import os
import cv2
import yaml
import numpy as np

import torch
import torch.nn.functional as F

from ltr.admin.environment import env_settings
from pytracking.tracker.base import BaseTracker

from ppq.api import quantize_onnx_model, export_ppq_graph, dump_torch_to_onnx, export_ppq_graph, quantize_torch_model, load_native_graph
from ppq import QuantizationSettingFactory, TargetPlatform
from ppq.executor.torch import TorchExecutor
from ppq.quantization.analyse.graphwise import graphwise_error_analyse

from ltr.admin.environment import env_settings as train_env

#from lib.utils.utils import load_yaml, im_to_torch, get_subwindow_tracking, make_scale_pyramid, python2round
def load_yaml(path, subset=True):
    file = open(path, 'r')
    yaml_obj = yaml.load(file.read(), Loader=yaml.FullLoader)

    if subset:
        hp = yaml_obj['TEST']
    else:
        hp = yaml_obj

    return hp

def to_torch(ndarray):
    return torch.from_numpy(ndarray)

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img

def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    """
    SiamFC type cropping
    """
    crop_info = dict()

    if isinstance(pos, float):
        pos = [pos, pos]

    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
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
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        # for return mask
        tete_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad))

        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        tete_im = np.zeros(im.shape[0:2])
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    crop_info['crop_cords'] = [context_xmin, context_xmax, context_ymin, context_ymax]
    crop_info['empty_mask'] = tete_im
    crop_info['pad_info'] = [top_pad, left_pad, r, c]

    if out_mode == "torch":
        return im_to_torch(im_patch.copy()), crop_info
    else:
        return im_patch, crop_info

def make_scale_pyramid(im, pos, in_side_scaled, out_side, avg_chans):
    """
    SiamFC 3/5 scale imputs
    """
    in_side_scaled = [round(x) for x in in_side_scaled]
    num_scale = len(in_side_scaled)
    pyramid = torch.zeros(num_scale, 3, out_side, out_side)
    max_target_side = in_side_scaled[-1]
    min_target_side = in_side_scaled[0]
    beta = out_side / min_target_side

    search_side = round(beta * max_target_side)
    search_region, _ = get_subwindow_tracking(im, pos, int(search_side), int(max_target_side), avg_chans, out_mode='np')

    for s, temp in enumerate(in_side_scaled):
        target_side = round(beta * temp)
        temp, _ = get_subwindow_tracking(search_region, (1 + search_side) / 2, out_side, target_side, avg_chans)
        pyramid[s, :] = temp
    return pyramid

def python2round(f):
    """
    use python2 round function in python3
    """
    if round(f + 1) - round(f) != 1:
        return f + abs(f) / f * 0.5
    return round(f)


class Lighttrack(BaseTracker):
    def __init__(self, params):
        super(Lighttrack, self).__init__(params)
        even = 0
        self.stride = 16
        self.even = even
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.params = params
        self.net = params.net.cuda()
        self.net.eval()
        self.info = params.info

        self.quant = self.params.quant

        self.initnet()

    def initialize(self, image, info: dict):
        x1,y1,w,h = info['init_bbox']
        target_pos = np.array((x1+w/2, y1+h/2),dtype=np.float64)
        target_sz = np.array((w, h),dtype=np.float64)
        ret_state = self.init(image, target_pos, target_sz, self.net)

        self.lighttrack_state = ret_state

        return {}

    def track(self, image, info: dict = None):
        ret_state = self._track(self.lighttrack_state, image)

        self.lighttrack_state = ret_state

        x,y = ret_state['target_pos']
        w,h = ret_state['target_sz']
        x1,y1 = x-w/2, y-h/2

        return {"target_bbox": (x1,y1,w,h)}

    def normalize(self, x):
        """ input is in (C,H,W) format"""
        x /= 255
        x -= self.mean
        x /= self.std
        return x

    def init(self, im, target_pos, target_sz, model):
        state = dict()

        p = Config(stride=self.stride, even=self.even)

        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]


        env = env_settings()
        yaml_path = os.path.join(env.pretrained_networks, 'LightTrack.yaml')

        cfg = load_yaml(yaml_path)
        cfg_benchmark = cfg['VOT2019']
        p.update(cfg_benchmark)
        p.renew()

        # 这里lighttrack模板的大小不固定，big_sz和small_sz不一样
        # 强制为 small_sz
        if False:
        #if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = cfg_benchmark['big_sz']
            p.renew()
        else:
            p.instance_size = cfg_benchmark['small_sz']
            p.renew()

        self.grids(p)  # self.grid_to_search_x, self.grid_to_search_y

        net = model

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))

        avg_chans = np.mean(im, axis=(0, 1))
        z_crop, _ = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
        z_crop = self.normalize(z_crop)
        z = z_crop.unsqueeze(0)
        z = z.cuda()
        #net.template(z.cuda())
        self.z_crop = z

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones(int(p.score_size), int(p.score_size))
        else:
            raise ValueError("Unsupported window type")

        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        return state

    def initnet(self):
        self.net.eval()
        if self.quant == False:
            pass
        else:
            # self.params.lock.acquire()
            env = train_env()
            QSetting = QuantizationSettingFactory.default_setting()
            QSetting.quantize_activation_setting.calib_algorithm = 'percentile'
            QSetting.quantize_parameter_setting.calib_algorithm = 'minmax'

            onnx_dir = os.path.join(env.workspace_dir, 'onnx', 'lighttrack')
            os.makedirs(onnx_dir, exist_ok=True)
            f_fp32 = os.path.join(onnx_dir, f'lighttrack_fp32.onnx')
            f_int8 = os.path.join(onnx_dir, f'lighttrack_int8.native')

            if os.path.exists(f_int8):
                quantized = load_native_graph(f_int8)
                print(f'load quant model from {f_int8}')
            else:
                dataset = []

                calibration_dir = os.path.join(env.workspace_dir, 'calibration', 'lighttrack')
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
                datalist = list(zip(zs.split(1, dim=0), xs.split(1, dim=0)))
                for z, x in datalist:
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
                class Wrapper(torch.nn.Module):
                    def __init__(self, net):
                        super(Wrapper, self).__init__()
                        self.net = net
                    def forward(self,z,x):
                        return self.net.trackzx(z,x)
                w = Wrapper(self.net)
                torch.onnx.export(w, (datasample,),
                                  f_fp32,
                                  input_names=['z', 'x'], output_names=['cls', 'reg'],
                                  # dynamic_axes={"z": [0], "x": [0], "tl_map": [0], "br_map": [0]},
                                  opset_version=13,
                                  keep_initializers_as_inputs=False,
                                  )
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

                # onnx check
                import onnxruntime
                sess = onnxruntime.InferenceSession(f_fp32, providers=['CUDAExecutionProvider'])
                input_feed = {'z': datasample['z'].cpu().numpy(), 'x': datasample['x'].cpu().numpy()}
                onnxruntime_outputs = sess.run(output_names=['cls', 'reg'], input_feed=input_feed)
                graphwise_error_analyse(graph=quantized, running_device=self.params.device,
                                        collate_fn=collate_fn,
                                        dataloader=analyse_dataset)

                export_ppq_graph(graph=quantized, platform=TargetPlatform.NATIVE,
                                 graph_save_to=f_int8)
                print(f'save quant model to {f_int8}')


            self.executor = TorchExecutor(quantized, device=self.params.device)

            # self.params.lock.release()
            pass

    def forwardnet(self,z,x):
        if self.quant == False:
            return self.net.trackzx(z,x)
        else:
            onnx_outputs = self.executor.forward({'z':z,'x':x},('cls','reg'))
            return onnx_outputs

    def update(self, net, x_crops, target_pos, target_sz, window, scale_z, p, debug=False):

        cls_score, bbox_pred = self.forwardnet(self.z_crop, x_crops)
        cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()

        # bbox to real predict
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # size penalty
        s_c = self.change(self.sz(pred_x2 - pred_x1, pred_y2 - pred_y1) / (self.sz_wh(target_sz)))  # scale penalty
        r_c = self.change((target_sz[0] / target_sz[1]) / ((pred_x2 - pred_x1) / (pred_y2 - pred_y1)))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * cls_score

        # window penalty
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence

        # get max
        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        # to real size
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1

        diff_xs = pred_xs - p.instance_size // 2
        diff_ys = pred_ys - p.instance_size // 2

        diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

        target_sz = target_sz / scale_z

        # size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr

        # size rate
        res_xs = target_pos[0] + diff_xs
        res_ys = target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]

        target_pos = np.array([res_xs, res_ys])
        target_sz = target_sz * (1 - lr) + lr * np.array([res_w, res_h])
        if debug:
            return target_pos, target_sz, cls_score[r_max, c_max], cls_score
        else:
            return target_pos, target_sz, cls_score[r_max, c_max]

    def _track(self, state, im):
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2  # slightly different from rpn++
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        x_crop, _ = get_subwindow_tracking(im, target_pos, p.instance_size, python2round(s_x), avg_chans)
        state['x_crop'] = x_crop.clone()  # torch float tensor, (3,H,W)
        x_crop = self.normalize(x_crop)
        x_crop = x_crop.unsqueeze(0)
        debug = True
        if debug:
            target_pos, target_sz, _, cls_score = self.update(net, x_crop.cuda(), target_pos, target_sz * scale_z,
                                                              window, scale_z, p, debug=debug)
            state['cls_score'] = cls_score
        else:
            target_pos, target_sz, _ = self.update(net, x_crop.cuda(), target_pos, target_sz * scale_z,
                                                   window, scale_z, p, debug=debug)
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['p'] = p

        return state

    def grids(self, p):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        # print('ATTENTION',p.instance_size,p.score_size)
        sz = p.score_size

        # the real shift is -param['shifts']
        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)


class Config(object):
    def __init__(self, stride=8, even=0):
        self.penalty_k = 0.062
        self.window_influence = 0.38
        self.lr = 0.765
        self.windowing = 'cosine'
        if even:
            self.exemplar_size = 128
            self.instance_size = 256
        else:
            self.exemplar_size = 127
            self.instance_size = 255
        # total_stride = 8
        # score_size = (instance_size - exemplar_size) // total_stride + 1 + 8  # for ++
        self.total_stride = stride
        self.score_size = int(round(self.instance_size / self.total_stride))
        self.context_amount = 0.5
        self.ratio = 0.94

    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        # self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1 + 8 # for ++
        self.score_size = int(round(self.instance_size / self.total_stride))
