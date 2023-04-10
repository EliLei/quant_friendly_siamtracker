# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import cv2
import torch
import time
import os

from ltr.models.pysot.core.config import cfg
from ltr.models.pysot.utils.anchor import Anchors
from pytracking.tracker.base import BaseTracker

from ppq.api import quantize_onnx_model, export_ppq_graph, dump_torch_to_onnx, export_ppq_graph, quantize_torch_model, load_native_graph
from ppq import QuantizationSettingFactory, TargetPlatform
from ppq.executor.torch import TorchExecutor
from ppq.quantization.analyse.graphwise import graphwise_error_analyse

from ltr.admin.environment import env_settings as train_env

class SiamRPNTracker(BaseTracker):
    def __init__(self, params):
        super(SiamRPNTracker, self).__init__(params)
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)

        self.params = params

        self.device = self.params.device
        self.net = params.net
        self.net = self.net.to(self.device)
        self.net.eval()

        self.quant = params.quant

        self.initnet()

    def initnet(self):
        self.net.eval()

        class Wrapper(torch.nn.Module):
            def __init__(self, net):
                super(Wrapper, self).__init__()
                self.net = net

            def forward(self, z, x):

                zf = self.net.backbone(z)
                if cfg.MASK.MASK:
                    zf = zf[-1]
                if cfg.ADJUST.ADJUST:
                    zf = self.net.neck(zf)
                zf = zf

                xf = self.net.backbone(x)
                if cfg.MASK.MASK:
                    self.xf = xf[:-1]
                    xf = xf[-1]
                if cfg.ADJUST.ADJUST:
                    xf = self.net.neck(xf)
                cls, loc = self.net.rpn_head(zf, xf)
                if cfg.MASK.MASK:
                    mask, self.net.mask_corr_feature = self.mask_head(zf, xf)
                    return cls, loc, mask

                return cls, loc

        self.net = Wrapper(self.net)
        self.net.eval()

        if self.quant == False:
            return
        env = train_env()
        QSetting = QuantizationSettingFactory.default_setting()
        QSetting.quantize_activation_setting.calib_algorithm = 'percentile'
        QSetting.quantize_parameter_setting.calib_algorithm = 'minmax'

        onnx_dir = os.path.join(env.workspace_dir, 'onnx', 'siamrpn')
        os.makedirs(onnx_dir, exist_ok=True)
        f_fp32 = os.path.join(onnx_dir, f'{self.params.nettype}_fp32.onnx')
        f_int8 = os.path.join(onnx_dir, f'{self.params.nettype}_int8.native')

        if os.path.exists(f_int8):
            quantized = load_native_graph(f_int8)
            print(f'load quant model from {f_int8}')
        else:
            dataset = []

            if self.params.nettype == 'siamrpn_alex_dwxcorr':
                calibration_dir = os.path.join(env.workspace_dir, 'calibration', 'siamrpn')
            elif self.params.nettype == 'siamrpn_mobilev2_l234_dwxcorr':
                calibration_dir = os.path.join(env.workspace_dir, 'calibration', 'siamrpn_mobilenet')
            z_dir = os.path.join(calibration_dir, 'z')
            x_dir = os.path.join(calibration_dir, 'x')
            i = 0
            zs = []
            xs = []
            while True:
                path_z = os.path.join(z_dir, f'{i:06}.data')
                path_x = os.path.join(x_dir, f'{i:06}.data')
                if not os.path.exists(path_z) or not os.path.exists(path_x):
                    break
                zs.append(torch.load(path_z))
                xs.append(torch.load(path_x))
                i += 1
            zs = torch.concat(zs, dim=0)
            xs = torch.concat(xs, dim=0)
            zs = zs*255
            xs =xs*255
            datalist = list(zip(zs.split(1, dim=0), xs.split(1, dim=0)))
            for z, x in datalist:
                # dataset.append({'z': z.to(self.params.device), 'x': x.to(self.params.device), 'zmask': zmask.to(self.params.device), 'xmask': xmask.to(self.params.device)})
                dataset.append({'z': z.to(self.params.device), 'x': x.to(self.params.device)})
            calibration_dataset = dataset[:len(dataset) // 2]
            analyse_dataset = dataset[len(dataset) // 2:]
            datasample = calibration_dataset[0]
            print(f"Calibration dataset loaded: len {len(calibration_dataset)}")
            print(f"Error analyse dataset loaded: len {len(analyse_dataset)}")
            BATCH_SIZE = 32

            def collate_fn(batch: dict) -> torch.Tensor:
                return {k: v.to(self.params.device) for k, v in batch.items()}

            # 这里 pytorch版本为 1.13， 导出时会有些参数作为indentity的输出，传递给conv，ppq无法处理conv动态参数，
            # 需要使用低版本 如 1.10
            torch.onnx.export(self.net, (datasample,),
                              f_fp32,
                              input_names=['z', 'x'], output_names=['cls','loc'],
                              # input_names=['z', 'x', 'zmask','xmask'], output_names=['pred'],
                              # dynamic_axes={"z": [0], "x": [0], "tl_map": [0], "br_map": [0]},
                              opset_version=13,
                              keep_initializers_as_inputs=False,
                              )
            # onnx check
            # import onnxruntime
            # sess = onnxruntime.InferenceSession(f_fp32, providers=['CUDAExecutionProvider'])
            # input_feed = {'z': datasample['z'].cpu().numpy(), 'x': datasample['x'].cpu().numpy()}
            # onnxruntime_outputs = sess.run(output_names=['pred'], input_feed=input_feed)
            # torch.onnx.export(torch.jit.trace(self.net,(datasample['z'],datasample['x'],datasample['zmask'],datasample['xmask'])), (datasample,),
            #                   f_fp32,
            #                   input_names=['z', 'x', 'zmask', 'xmask'], output_names=['pred'],
            #                   # dynamic_axes={"z": [0], "x": [0], "tl_map": [0], "br_map": [0]},
            #                   opset_version=13,
            #                   keep_initializers_as_inputs=False,
            #                   )
            quantized = quantize_onnx_model(f_fp32, calibration_dataset,
                                            calib_steps=len(calibration_dataset) // BATCH_SIZE,
                                            input_shape=None, platform=TargetPlatform.SNPE_INT8,
                                            setting=QSetting, collate_fn=collate_fn,
                                            inputs=datasample, device=self.params.device)

            # quantized = quantize_torch_model(self.net, calibration_dataset, calib_steps=len(calibration_dataset)//BATCH_SIZE,
            #                                  input_shape=None, platform=TargetPlatform.SNPE_INT8,
            #                                  setting=QSetting, collate_fn=collate_fn,
            #                                  onnx_export_file=f_fp32,
            #                                  inputs=datasample, device=self.params.device)

            # graphwise_error_analyse(graph=quantized, running_device=self.params.device,
            #                         collate_fn=collate_fn,
            #                         dataloader=analyse_dataset)

            export_ppq_graph(graph=quantized, platform=TargetPlatform.NATIVE,
                             graph_save_to=f_int8)
            print(f'save quant model to {f_int8}')

        self.executor = TorchExecutor(quantized, device=self.params.device)

        # self.params.lock.release()
        pass

    def forwardnet(self,z,x, forcefp=False):
        if self.quant == False or forcefp:
            cls, loc = self.net(z,x)
        else:
            cls, loc = self.executor.forward({'z': z, 'x': x}, ['cls','loc'])
            #print(cls,loc)

        return {
                'cls': cls,
                'loc': loc,
               }


    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        if cfg.CUDA:
            im_patch = im_patch.cuda()
        return im_patch


    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def initialize(self, image, info: dict):
        tic = time.time()

        bbox = info['init_bbox']

        self.init(image, bbox)

        out = {'time': time.time() - tic}

    def track(self, image, info: dict = None):
        ret = self._track(image)

        return {"target_bbox": ret['bbox']}


    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        #self.net.template(z_crop)
        self.z = z_crop

    def _track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        #outputs = self.net.track(x_crop)
        outputs = self.forwardnet(self.z, x_crop)
        # outputs2 = self.forwardnet(self.z, x_crop, forcefp=True)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)
        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score
        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)



        # score2 = self._convert_score(outputs2['cls'])
        # pred_bbox2 = self._convert_bbox(outputs2['loc'], self.anchors)
        # # scale penalty
        # s_c = change(sz(pred_bbox2[2, :], pred_bbox2[3, :]) /
        #              (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
        # # aspect ratio penalty
        # r_c = change((self.size[0] / self.size[1]) /
        #              (pred_bbox2[2, :] / pred_bbox2[3, :]))
        # penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        # pscore2 = penalty * score2
        # # window penalty
        # pscore2 = pscore2 * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
        #          self.window * cfg.TRACK.WINDOW_INFLUENCE
        # best_idx2 = np.argmax(pscore2)
        #
        # print(best_idx, best_idx2,score[best_idx],score2[best_idx2])
        # print(((score-score2)**2/score2**2).mean())
        # print(((pred_bbox - pred_bbox2) ** 2 / pred_bbox2 ** 2).mean())

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }
