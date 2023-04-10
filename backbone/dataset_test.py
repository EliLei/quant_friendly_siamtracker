from imagenet_dataset import ImageNet
import torchvision.transforms as transforms
import torchvision
import random
import numpy as np
import os

from typing import Iterable

import torch
import torchvision
from torchvision import transforms
from ppq import TorchExecutor
from ppq.api import quantize_torch_model
from ppq.quantization.analyse.graphwise import statistical_analyse
from ppq.quantization.analyse.layerwise import layerwise_error_analyse
from ppq import QuantableOperation, TargetPlatform, graphwise_error_analyse
from ppq.api import quantize_torch_model
from ppq.api.interface import (ENABLE_CUDA_KERNEL, dispatch_graph,
                               dump_torch_to_onnx, load_onnx_graph, quantize_native_model)
from ppq.api.setting import QuantizationSettingFactory
from torch.utils.data import DataLoader
import lightning as pl
import timm
import numpy as np
import onnxruntime

from matplotlib import pyplot as plt

BATCHSIZE = 32
INPUT_SHAPE = [BATCHSIZE, 3, 224, 224]
DEVICE = 'cuda'
PLATFORM = TargetPlatform.SNPE_INT8
basedir = os.path.split(__file__)[0]

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])


dataset = ImageNet(transform = train_transform)
dataset_val = ImageNet(split='val',transform = train_transform)

np.random.seed(212527)
post_quantization_idx = np.random.randint(0,len(dataset),32*32)
post_analyse_idx = np.random.randint(0,len(dataset),32*32)
def load_calibration_dataset() -> Iterable:
    # ------------------------------------------------------------
    # 让我们从创建 calibration 数据开始做起， PPQ 需要你送入 32 ~ 1024 个样本数据作为校准数据集
    # 它们应该尽可能服从真实样本的分布，量化过程如同训练过程一样存在可能的过拟合问题
    # 你应当保证校准数据是经过正确预处理的、有代表性的数据，否则量化将会失败；校准数据不需要标签；数据集不能乱序
    # ------------------------------------------------------------
    # return [torch.rand(size=INPUT_SHAPE) for _ in range(32)]
    return [torch.stack([dataset[post_quantization_idx[i + j * 32]][0] for i in range(32)]) for j in range(32)]
    #return [torch.randn(INPUT_SHAPE) for j in range(32)]

def load_calibration_dataset2() -> Iterable:
    # ------------------------------------------------------------
    # 让我们从创建 calibration 数据开始做起， PPQ 需要你送入 32 ~ 1024 个样本数据作为校准数据集
    # 它们应该尽可能服从真实样本的分布，量化过程如同训练过程一样存在可能的过拟合问题
    # 你应当保证校准数据是经过正确预处理的、有代表性的数据，否则量化将会失败；校准数据不需要标签；数据集不能乱序
    # ------------------------------------------------------------
    # return [torch.rand(size=INPUT_SHAPE) for _ in range(32)]
    return [torch.stack([dataset[post_analyse_idx[i + j * 32]][0] for i in range(32)]) for j in range(32)]

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)
CALIBRATION = load_calibration_dataset()
CALIBRATION2 = load_calibration_dataset2()


# model = torchvision.models.shufflenet_v2_x1_0(pretrained=True).to(DEVICE)
#
# torch.onnx.export(model, CALIBRATION[0].to(DEVICE), os.path.join(basedir, f'shufflenet_v2_x1_0.onnx'), export_params=True, verbose=False, input_names=('input0'),
# output_names=('output0'),opset_version=11,dynamic_axes={'input0':[0],'output0':[0]})
# onnx_model = onnxruntime.InferenceSession(os.path.join(basedir, f'shufflenet_v2_x1_0.onnx'), providers=['CUDAExecutionProvider'])
# data = np.random.rand(1,*INPUT_SHAPE[1:]).astype(np.float32)
# onnx_input = {onnx_model.get_inputs()[0].name: data}
# outputs = onnx_model.run(None, onnx_input)
#
#
# QSetting = QuantizationSettingFactory.default_setting()
# QSetting.quantize_activation_setting.calib_algorithm = 'mse'
# QSetting.quantize_parameter_setting.calib_algorithm = 'minmax'
# quantized = quantize_torch_model(
#     model=model, calib_dataloader=CALIBRATION,
#     calib_steps=32, input_shape=INPUT_SHAPE,
#     collate_fn=collate_fn, platform=PLATFORM,
#     device=DEVICE, verbose=0, setting=QSetting,
#     onnx_export_file=os.path.join(basedir, f'shufflenet_v2_x1_0_quan.onnx'))
#
# onnx_model = onnxruntime.InferenceSession(os.path.join(basedir, f'shufflenet_v2_x1_0_quan.onnx'), providers=['CUDAExecutionProvider'])
# data = np.random.rand(1,*INPUT_SHAPE[1:]).astype(np.float32)
# onnx_input = {onnx_model.get_inputs()[0].name: data}
# outputs = onnx_model.run(None, onnx_input)
# print(outputs)
# executor = TorchExecutor(graph=quantized, fp16_mode=False, device=DEVICE)
# out = executor.forward( torch.randn(1,*INPUT_SHAPE[1:]).to(DEVICE) )

def model_factor(model_name):
    if model_name=='mobilenet_v2':
        return torchvision.models.mobilenet_v2(pretrained=True)
    elif model_name=='mobilenet_v3':
        return torchvision.models.mobilenet_v3_small(pretrained=True)
    elif model_name=='shufflenet_v2':
        return torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    elif model_name=='resnet18':
        return torchvision.models.resnet18(pretrained=True)
    elif model_name=='resnet50':
        return torchvision.models.resnet50(pretrained=True)
    elif model_name=='swin':
        return torchvision.models.swin_t(weights='DEFAULT')
    elif model_name=='swinv2':
        return torchvision.models.swin_v2_t(weights='DEFAULT')
    elif model_name=='vit':
        return torchvision.models.vit_b_16(weights='DEFAULT')
    raise NotImplementedError(f'Error nettype {nettype}')


# quantization accuracy drop
if True:

    class Wrapper_pl(pl.LightningModule):
        def __init__(self, nettype,quantization=False):
            super(Wrapper_pl, self).__init__()
            self.model = model_factor(nettype).to(DEVICE)
            self.quantization = quantization
            if quantization:
                self.eval()
                self.model.eval()
                # PPQ 提供 kl, mse, minmax, isotone, percentile(默认) 五种校准方法
                # 每一种校准方法还有更多参数可供调整，PPQ 也允许你单独调整某一层的量化校准方法
                # 在这里我们首先展示以 QSetting 的方法调整量化校准参数(推荐)
                QSetting = QuantizationSettingFactory.default_setting()
                QSetting.quantize_activation_setting.calib_algorithm = 'percentile'
                QSetting.quantize_parameter_setting.calib_algorithm = 'minmax'
                # 更进一步地，当你选择了某种校准方法，你可以进入 ppq.core.common
                # OBSERVER_KL_HIST_BINS, OBSERVER_PERCENTILE, OBSERVER_MSE_HIST_BINS 皆是与校准方法相关的可调整参数
                # OBSERVER_KL_HIST_BINS - KL 算法相关的箱子个数，你可以试试将其调整为 512, 1024, 2048, 4096, 8192 ...
                # OBSERVER_PERCENTILE - Percentile 算法相关的百分比，你可以试试将其调整为 0.9999, 0.9995, 0.99999, 0.99995 ...
                # OBSERVER_MSE_HIST_BINS - MSE 算法相关的箱子个数，你可以试试将其调整为 512, 1024, 2048, 4096, 8192 ...

                self.quantized = quantize_torch_model(
                    model=self.model, calib_dataloader=CALIBRATION,
                    calib_steps=32, input_shape=INPUT_SHAPE,
                    collate_fn=collate_fn, platform=PLATFORM,
                    device=DEVICE, verbose=0, setting=QSetting,
                    onnx_export_file=os.path.join(basedir, f'{nettype}.onnx'))

                self.executor = [TorchExecutor(graph=self.quantized, fp16_mode=False, device=DEVICE)]


        def forward(self, x):
            if not self.quantization:
                return self.model(x)
            return self.executor[0].forward(x)[0]

        def test_step(self,test_batch, batch_idx):
            x,y = test_batch
            pred = self(x)
            pred = torch.argmax(pred,dim=1)


            test_acc = torch.sum(y == pred).item() / (y.shape[0] * 1.0)
            self.log('test_acc', test_acc)



    dataloader_val = DataLoader(dataset_val,batch_size=BATCHSIZE, num_workers=8)
    trainer = pl.Trainer(gpus=1)

    #for nettype in [ 'mobilenet_v2', 'mobilenet_v3', 'resnet18', 'resnet50', 'vit', 'swin', 'swinv2']:
    for nettype in ['mobilenet_v2']:
        try:
            print("*************************")
            print(nettype)
            print("*************************")

            model = Wrapper_pl(nettype, quantization=False)
            model_q = Wrapper_pl(nettype, quantization=True)

            reports = graphwise_error_analyse(
                graph=model_q.quantized, running_device=DEVICE, collate_fn=collate_fn,
                dataloader=CALIBRATION)

            trainer.test(model_q, dataloader_val)
            trainer.test(model, dataloader_val)
        except:
            print(f'skip {nettype}')