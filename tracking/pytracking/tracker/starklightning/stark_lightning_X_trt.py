#from ltr.data.processing_utils import sample_target
from ltr.admin.environment import env_settings as train_env
from ltr.models.stark.utils.box_ops import clip_box
from ltr.models.stark.utils.merge import get_qkv
from pytracking.tracker.base import BaseTracker
import torch
import numpy as np
import os
import time
import cv2
from ppq.api import quantize_onnx_model, export_ppq_graph, dump_torch_to_onnx, export_ppq_graph, quantize_torch_model, load_native_graph
from ppq import QuantizationSettingFactory, TargetPlatform
from ppq.executor.torch import TorchExecutor
from ppq.quantization.analyse.graphwise import graphwise_error_analyse

import math
import torch.nn.functional as F
import cv2 as cv

#stark 修改版的sample_target， 与pytracking不一样， 免得出错，就搬过来了
def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H,W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded

class PreprocessorX(object):
    def __init__(self,device='cuda'):
        self.device=device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).to(device)

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).to(self.device).float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        #amask_tensor = amask_arr.to(torch.bool).to(self.device).unsqueeze(dim=0)  # (1,H,W)
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return img_tensor_norm, amask_tensor

class STARK_LightningXtrt(BaseTracker):
    def __init__(self, params):
        super(STARK_LightningXtrt, self).__init__(params)

        self.params = params
        self.net = self.params.net
        self.net.eval()
        self.device = params.device
        self.net = self.net.to(self.device)
        self.quant = self.params.quant



        self.cfg = params.cfg

        self.preprocessor = PreprocessorX()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        self.z_dict1 = {}

        self.initnet()

    def initnet(self):
        self.net.eval()
        # if self.quant == False:
        #     return

        class Wrapper(torch.nn.Module):
            def __init__(self, net):
                super(Wrapper, self).__init__()
                self.net = net

            #def forward(self,z,x):
            def forward(self, z, x, zmask, xmask):
                zdict = self.net.forward_backbone(z, zx="template0", mask=zmask)
                xdict = self.net.forward_backbone(x, zx="search", mask=xmask)

                #zdict = self.net.forward_backbone(z, zx="template0", mask=None)
                #xdict = self.net.forward_backbone(x, zx="search", mask=None)

                # {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}
                q = xdict["feat"]+xdict["pos"]
                k = torch.cat([zdict['feat'],xdict["feat"]],dim=1)+torch.cat([zdict['pos'],xdict["pos"]],dim=1)
                v = torch.cat([zdict['feat'],xdict["feat"]],dim=1)
                key_padding_mask = torch.cat([zdict['mask'],xdict["mask"]],dim=1).unsqueeze(-1)
                key_padding_mask = 1-key_padding_mask
                k=k*key_padding_mask
                v=v*key_padding_mask
                # run the transformer
                out_dict, _, _ = self.net.forward_transformer(q=q, k=k, v=v)
                #out_dict, _, _ = self.net.forward_transformer(q=q, k=k, v=v, key_padding_mask=key_padding_mask)
                pred = out_dict['pred_boxes']

                return pred
        self.net = Wrapper(self.net)
        self.net.eval()
        if self.quant == False:
            return
        env = train_env()
        QSetting = QuantizationSettingFactory.default_setting()
        QSetting.quantize_activation_setting.calib_algorithm = 'percentile'
        QSetting.quantize_parameter_setting.calib_algorithm = 'minmax'

        onnx_dir = os.path.join(env.workspace_dir, 'onnx', 'starklightning')
        os.makedirs(onnx_dir, exist_ok=True)
        f_fp32 = os.path.join(onnx_dir, f'starklightning_fp32.onnx')
        f_int8 = os.path.join(onnx_dir, f'starklightning_int8.native')

        if os.path.exists(f_int8):
            quantized = load_native_graph(f_int8)
            print(f'load quant model from {f_int8}')
        else:
            dataset = []

            calibration_dir = os.path.join(env.workspace_dir, 'calibration', 'starklightning')
            z_dir = os.path.join(calibration_dir, 'z')
            x_dir = os.path.join(calibration_dir, 'x')
            zmask_dir = os.path.join(calibration_dir, 'zmask')
            xmask_dir = os.path.join(calibration_dir, 'xmask')
            i = 0
            zs = []
            xs = []
            zmasks = []
            xmasks = []
            while True:
                path_z = os.path.join(z_dir, f'{i:06}.data')
                path_x = os.path.join(x_dir, f'{i:06}.data')
                path_zmask = os.path.join(zmask_dir, f'{i:06}.data')
                path_xmask = os.path.join(xmask_dir, f'{i:06}.data')
                if not os.path.exists(path_z) or not os.path.exists(path_x) or not os.path.exists(path_zmask) or not os.path.exists(path_xmask):
                    break
                zs.append(torch.load(path_z))
                xs.append(torch.load(path_x))
                zmasks.append(torch.load(path_zmask))
                xmasks.append(torch.load(path_xmask))
                i += 1
            zs = torch.concat(zs, dim=0)
            xs = torch.concat(xs, dim=0)
            zmasks = torch.concat(zmasks, dim=0).to(torch.float)
            xmasks = torch.concat(xmasks, dim=0).to(torch.float)
            datalist = list(zip(zs.split(1, dim=0), xs.split(1, dim=0),zmasks.split(1, dim=0),xmasks.split(1, dim=0)))
            for z, x ,zmask, xmask in datalist:
                dataset.append({'z': z.to(self.params.device), 'x': x.to(self.params.device), 'zmask': zmask.to(self.params.device), 'xmask': xmask.to(self.params.device)})
                #dataset.append({'z': z.to(self.params.device), 'x': x.to(self.params.device)})
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
                              #input_names=['z', 'x'], output_names=['pred'],
                              input_names=['z', 'x', 'zmask','xmask'], output_names=['pred'],
                              # dynamic_axes={"z": [0], "x": [0], "tl_map": [0], "br_map": [0]},
                              opset_version=13,
                              keep_initializers_as_inputs=False,
                              )
            #onnx check
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


            graphwise_error_analyse(graph=quantized, running_device=self.params.device,
                                    collate_fn=collate_fn,
                                    dataloader=analyse_dataset)

            export_ppq_graph(graph=quantized, platform=TargetPlatform.NATIVE,
                             graph_save_to=f_int8)
            print(f'save quant model to {f_int8}')


        self.executor = TorchExecutor(quantized, device=self.params.device)

        # self.params.lock.release()
        pass

    def forwardnet(self,z,x,zmask,xmask):
        if self.quant == False:
            return self.net(z, x, zmask, xmask)
            with torch.no_grad():
                zdict = self.net.net.forward_backbone(z, zx="template0", mask=zmask)
                xdict = self.net.net.forward_backbone(x, zx="search", mask=xmask)

                # {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}
                q = xdict["feat"] + xdict["pos"]
                k = torch.cat([zdict['feat'], xdict["feat"]], dim=1) + torch.cat([zdict['pos'], xdict["pos"]], dim=1)
                v = torch.cat([zdict['feat'], xdict["feat"]], dim=1)
                key_padding_mask = torch.cat([zdict['mask'],xdict["mask"]],dim=1)
                # run the transformer
                out_dict, _, _ = self.net.net.forward_transformer(q=q, k=k, v=v, key_padding_mask=key_padding_mask)
                ret = out_dict['pred_boxes']

                ret2 = self.net(z, x, zmask, xmask)
                print((ret-ret2)**2/(ret**2))
            return ret
        else:
            #onnx_outputs = self.executor.forward({'z': z, 'x': x}, ('pred',))
            onnx_outputs = self.executor.forward({'z': z, 'x': x,'zmask':zmask,'xmask':xmask}, ('pred',))

            return onnx_outputs[0]



    def initialize(self, image, info: dict):
        # forward the template once
        H, W, C = image.shape
        #z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
        #                                            output_sz=self.params.template_size, mask=torch.ones((H,W),dtype=torch.float))
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        template, template_mask = self.preprocessor.process(z_patch_arr, z_amask_arr)
        template_mask = template_mask.to(torch.float)
        #template_mask = torch.logical_not(template_mask)
        #with torch.no_grad():
        #    self.z_dict1 = self.net.forward_backbone(template, zx="template0", mask=template_mask)
        self.z = template
        self.zmask = template_mask
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        # x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
        #                                                         output_sz=self.params.search_size,mask=torch.ones((H,W),dtype=torch.float))  # (x1, y1, w, h)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search, search_mask = self.preprocessor.process(x_patch_arr, x_amask_arr)
        search_mask = search_mask.to(torch.float)
        # #search_mask = torch.logical_not(search_mask)
        # with torch.no_grad():
        #     x_dict = self.net.forward_backbone(search, zx="search", mask=search_mask)
        #     # merge the template and the search
        #     feat_dict_list = [self.z_dict1, x_dict]
        #     q, k, v, key_padding_mask = get_qkv(feat_dict_list)
        #     # run the transformer
        #     out_dict, _, _ = self.net.forward_transformer(q=q, k=k, v=v, key_padding_mask=key_padding_mask)
        #pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        x = search
        xmask = search_mask
        pred_boxes = self.forwardnet(self.z,x,self.zmask,xmask)
        pred_boxes = pred_boxes.view(-1, 4)

        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


