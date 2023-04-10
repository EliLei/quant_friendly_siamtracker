from ltr.admin.environment import env_settings as train_env
from ltr.data.processing_utils import sample_target
from pytracking.tracker.base import BaseTracker
import torch
import numpy as np
import os
import time

from ppq.api import quantize_onnx_model, export_ppq_graph, dump_torch_to_onnx, export_ppq_graph, quantize_torch_model, load_native_graph
from ppq import QuantizationSettingFactory, TargetPlatform
from ppq.executor.torch import TorchExecutor
from ppq.quantization.analyse.graphwise import graphwise_error_analyse
from torch.utils.data import DataLoader

class QFNET(BaseTracker):

    def __init__(self, params):
        super(QFNET, self).__init__(params)

        self.frame_num = 1
        self.quant = self.params.quant

        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'
        self.net = self.params.net.to(self.params.device)

        self.initialize_net()

        self.state = None

        self.pp_mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.pp_std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def preprocess(self, img_arr: np.ndarray):

        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).to(self.params.device).float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.pp_mean) / self.pp_std  # (1,3,H,W)
        # Deal with the attention mask

        return img_tensor_norm

    def initialize_net(self):
        self.net.eval()
        if not self.quant:
            pass
        else:
            #self.params.lock.acquire()

            env = train_env()
            QSetting = QuantizationSettingFactory.default_setting()
            QSetting.quantize_activation_setting.calib_algorithm = 'percentile'
            QSetting.quantize_parameter_setting.calib_algorithm = 'minmax'

            platform = self.params.target_platform

            onnx_dir = os.path.join(env.workspace_dir, 'onnx', 'qfnet', self.params.nettype)
            os.makedirs(onnx_dir, exist_ok=True)
            if platform==TargetPlatform.SNPE_INT8:
                f_fp32 = os.path.join(onnx_dir, f'qfnet_{self.params.nettype}_fp32.onnx')
                f_int8 = os.path.join(onnx_dir, f'qfnet_{self.params.nettype}_int8.native')
            else:
                f_fp32 = os.path.join(onnx_dir, f'qfnet_{self.params.nettype}_{platform}_fp32.onnx')
                f_int8 = os.path.join(onnx_dir, f'qfnet_{self.params.nettype}_{platform}_int8.native')

            if os.path.exists(f_int8):
                quantized = load_native_graph(f_int8)
                print(f'load quant model from {f_int8}')
            else:
                dataset = []

                if self.params.template_size == 144 and self.params.search_size == 304:
                    calibration_dir = os.path.join(env.workspace_dir, 'calibration', '144_304')
                else:
                    calibration_dir = os.path.join(env.workspace_dir, 'calibration', 'qfnet')
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


                if self.net.corner_head is not None:
                    # 这里 pytorch版本为 1.13， 导出时会有些参数作为indentity的输出，传递给conv，ppq无法处理conv动态参数，
                    # 需要使用低版本 如 1.10
                    torch.onnx.export(self.net, (datasample,),
                                      f_fp32,
                                      input_names =['z','x'], output_names=['tl_map', 'br_map'],
                                      #dynamic_axes={"z": [0], "x": [0], "tl_map": [0], "br_map": [0]},
                                      opset_version=13,
                                      keep_initializers_as_inputs=False,
                                      )
                    quantized = quantize_onnx_model(f_fp32, calibration_dataset, calib_steps=len(calibration_dataset)//BATCH_SIZE,
                                        input_shape=None, platform=platform,
                                        setting=QSetting, collate_fn=collate_fn,
                                        inputs=datasample, device=self.params.device)

                    # quantized = quantize_torch_model(self.net, calibration_dataset, calib_steps=len(calibration_dataset)//BATCH_SIZE,
                    #                                  input_shape=None, platform=TargetPlatform.SNPE_INT8,
                    #                                  setting=QSetting, collate_fn=collate_fn,
                    #                                  onnx_export_file=f_fp32,
                    #                                  inputs=datasample, device=self.params.device)

                    # onnx check
                    # import onnxruntime
                    # sess = onnxruntime.InferenceSession(f_fp32, providers=['CUDAExecutionProvider'])
                    # input_feed = {'z': datasample['z'].cpu().numpy(), 'x': datasample['x'].cpu().numpy()}
                    # onnxruntime_outputs = sess.run(output_names=['tl_map', 'br_map'], input_feed=input_feed)
                    # graphwise_error_analyse(graph=quantized, running_device=self.params.device,
                    #                         collate_fn=collate_fn,
                    #                         dataloader=analyse_dataset)

                    export_ppq_graph(graph=quantized, platform=TargetPlatform.NATIVE,
                                     graph_save_to=f_int8)
                    print(f'save quant model to {f_int8}')

                elif self.net.bbox_head is not None:
                    # 这里 pytorch版本为 1.13， 导出时会有些参数作为indentity的输出，传递给conv，ppq无法处理conv动态参数，
                    # 需要使用低版本 如 1.10
                    torch.onnx.export(self.net, (datasample,),
                                      f_fp32,
                                      input_names=['z', 'x'], output_names=['x1y1x2y2'],
                                      # dynamic_axes={"z": [0], "x": [0], "tl_map": [0], "br_map": [0]},
                                      opset_version=13,
                                      keep_initializers_as_inputs=False,
                                      )
                    quantized = quantize_onnx_model(f_fp32, calibration_dataset,
                                                    calib_steps=len(calibration_dataset) // BATCH_SIZE,
                                                    input_shape=None, platform=platform,
                                                    setting=QSetting, collate_fn=collate_fn,
                                                    inputs=datasample, device=self.params.device)

                    # quantized = quantize_torch_model(self.net, calibration_dataset, calib_steps=len(calibration_dataset)//BATCH_SIZE,
                    #                                  input_shape=None, platform=TargetPlatform.SNPE_INT8,
                    #                                  setting=QSetting, collate_fn=collate_fn,
                    #                                  onnx_export_file=f_fp32,
                    #                                  inputs=datasample, device=self.params.device)

                    # onnx check
                    # import onnxruntime
                    # sess = onnxruntime.InferenceSession(f_fp32, providers=['CUDAExecutionProvider'])
                    # input_feed = {'z': datasample['z'].cpu().numpy(), 'x': datasample['x'].cpu().numpy()}
                    # onnxruntime_outputs = sess.run(output_names=['x1y1x2y2'], input_feed=input_feed)
                    # graphwise_error_analyse(graph=quantized, running_device=self.params.device,
                    #                         collate_fn=collate_fn,
                    #                         dataloader=analyse_dataset)

                    export_ppq_graph(graph=quantized, platform=TargetPlatform.NATIVE,
                                     graph_save_to=f_int8)
                    print(f'save quant model to {f_int8}')
                else:
                    raise NotImplementedError


            self.executor = TorchExecutor(quantized, device=self.params.device)

            #self.params.lock.release()
            pass

    def forward_net(self,z,x):
        if not self.quant:
            with torch.no_grad():
                return self.net.get_bb(z,x)
        else:
            if self.net.corner_head is not None:
                onnx_outputs = self.executor.forward({'z':z,'x':x},('tl_map','br_map'))
            elif self.net.bbox_head is not None:
                onnx_outputs = self.executor.forward({'z': z, 'x': x}, ('x1y1x2y2',))

            #import matplotlib.pyplot as plt;plt.figure('tl');plt.imshow(onnx_outputs[0].detach().cpu().numpy()[0,0]);plt.figure('br');plt.imshow(onnx_outputs[1].detach().cpu().numpy()[0,0]);plt.show()

            return self.net.postprocess_bb(onnx_outputs)


    def initialize(self, image, info: dict):
        # forward the template once
        self.frame_num = 1

        tic = time.time()

        z_patch_arr, _ = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z = self.preprocess(z_patch_arr)
        self.state = info['init_bbox']


        out = {'time': time.time() - tic}

        return out


    def track(self, image, info: dict = None):
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        H, W, _ = image.shape

        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x = self.preprocess(x_patch_arr)

        out = self.forward_net(self.z, x)
        pred_box = out.view(-1,4)[0].detach().cpu().numpy() # x1 y1 x2 y2 norm

        pred_box = (pred_box * self.params.search_size / resize_factor).tolist()  # x1 y1 x2 y2 对于crop出来的patch的坐标
        # get the final box result
        self.state = self.clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_p, cy_p = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        x1_c, y1_c, x2_c, y2_c = pred_box
        cx_c, cy_c, w_c, h_c  = (x1_c+x2_c)/2, (y1_c+y2_c)/2, abs(x2_c-x1_c), abs(y2_c-y1_c)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx_c + (cx_p - half_side)
        cy_real = cy_c + (cy_p - half_side)
        return [cx_real - 0.5 * w_c, cy_real - 0.5 * h_c, w_c, h_c]

    def clip_box(self, box: list, H, W, margin=0):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        x1 = min(max(0, x1), W - margin)
        x2 = min(max(margin, x2), W)
        y1 = min(max(0, y1), H - margin)
        y2 = min(max(margin, y2), H)
        w = max(margin, x2 - x1)
        h = max(margin, y2 - y1)
        return [x1, y1, w, h]