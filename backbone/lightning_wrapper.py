import os
import time


from backbone_models import backbone_factory
from imagenet_dataset import ImageNet

import pytorch_lightning as pl

import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.utils.data

from torchinfo import summary

import numpy as np
import onnx


from ppq.api import quantize_onnx_model, export_ppq_graph
from ppq import QuantizationSettingFactory, TargetPlatform
from ppq.executor.torch import TorchExecutor
from ppq.quantization.analyse.graphwise import graphwise_error_analyse

def ImageNetDataModule(batch_size=256,num_workers=8):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    train_dataset = ImageNet(split='train', transform=train_transform)
    val_dataset = ImageNet(split='val', transform=test_transform)
    test_dataset = ImageNet(split='val', transform=test_transform)
    return pl.LightningDataModule.from_datasets(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset,
                                                batch_size=batch_size, num_workers=num_workers)

#
# class ImageNetDataModule(pl.LightningDataModule):
#     def __init__(self, batch_size=256):
#         self.batch_size = batch_size
#         self.prepare_data_per_node = False
#
#     def prepare_data(self):
#         pass
#
#     def setup(self, stage):
#         train_transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                  std=(0.229, 0.224, 0.225))
#         ])
#         test_transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                  std=(0.229, 0.224, 0.225))
#         ])
#         # Assign train/val datasets for use in dataloaders
#         if stage == "fit":
#             self.train_dataset = ImageNet(split='train', transform=train_transform)
#             self.val_dataset = ImageNet(split='val', transform=train_transform)
#         elif stage == 'validate':
#             self.val_dataset = ImageNet(split='val', transform=train_transform)
#         # Assign test dataset for use in dataloader(s)
#         elif stage == "test":
#             self.test_dataset = ImageNet(split='val', transform=test_transform)
#         else:
#             raise NotImplementedError
#
#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
#
#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
#
#     def test_dataloader(self):
#         return torch.utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=8)

class ModelWrapper(pl.LightningModule):
    def __init__(self, nettype, width_mult=1.0, input_size=(3, 224, 224), epoch=50):
        super(ModelWrapper, self).__init__()
        self.save_hyperparameters('nettype', 'width_mult', 'input_size','epoch')

        self.nettype = nettype
        self.width_mult = width_mult
        self.input_size = tuple(input_size)
        self.epoch=epoch

        self.net = backbone_factory(nettype, width_mult)

        self.quantization = False

    def quantize(self, device):
        self.eval()

        f_fp32 = os.path.join(os.path.split(__file__)[0],'logs', f'{self.netname()}_fp32.onnx')
        f_int8 = os.path.join(os.path.split(__file__)[0],'logs', f'{self.netname()}_int8.native')
        data = torch.randn(1,*self.input_size)
        torch.onnx.export(self.net, data, f_fp32,
                          input_names=('input0',),
                          output_names=('output0',),
                          opset_version=13,
                          #dynamic_axes={'input0': [0], 'output0': [0]},
                          keep_initializers_as_inputs=False,
                          )
        onnx.shape_inference.infer_shapes_path(f_fp32, f_fp32)


        def load_calibration_dataset(error_analyse=False):
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])
            dataset = ImageNet(split='train', transform=train_transform)

            if error_analyse == False:
                np.random.seed(212527)
            else:
                np.random.seed(123)
            post_quantization_idx = np.random.randint(0, len(dataset), 32 * 32)
            np.random.seed(int(time.time()*1000)&0xffffffff)

            return [torch.stack([dataset[post_quantization_idx[i + j * 32]][0] for i in range(32)]) for j in range(32)]
            # return [torch.randn(INPUT_SHAPE) for j in range(32)]

        def collate_fn(batch: torch.Tensor) -> torch.Tensor:
            return batch.to(device)

        if getattr(ModelWrapper,'CALIBRATION',None) is None:
            ModelWrapper.CALIBRATION = {}
        if self.input_size not in ModelWrapper.CALIBRATION:
            ModelWrapper.CALIBRATION[self.input_size] = load_calibration_dataset()

        if getattr(ModelWrapper,'ERROR_ANALYSE',None) is None:
            ModelWrapper.ERROR_ANALYSE = {}
        if self.input_size not in ModelWrapper.ERROR_ANALYSE:
            ModelWrapper.ERROR_ANALYSE[self.input_size] = load_calibration_dataset(error_analyse=True)

        QSetting = QuantizationSettingFactory.default_setting()
        QSetting.quantize_activation_setting.calib_algorithm = 'percentile'
        QSetting.quantize_parameter_setting.calib_algorithm = 'minmax'

        quantized = quantize_onnx_model(f_fp32, ModelWrapper.CALIBRATION[self.input_size], calib_steps=32, input_shape=(1,*self.input_size), collate_fn=collate_fn,
                            setting=QSetting, platform=TargetPlatform.SNPE_INT8, device=device, verbose=0)
        self.executor = [TorchExecutor(quantized, device=device),]
        graphwise_error_analyse(graph=quantized, running_device=device, collate_fn=collate_fn, dataloader=ModelWrapper.ERROR_ANALYSE[self.input_size])
        # export_ppq_graph(graph=quantized, platform=TargetPlatform.NATIVE,
        #                  graph_save_to=f_int8)
        # export_ppq_graph(graph=quantized, platform=TargetPlatform.SNPE_INT8,
        #                  graph_save_to=os.path.splitext(f_int8)[0])

        pass

    def summary(self):
        summary(self.net, input_size=(1,*self.input_size),device='cuda')
        pass

    def netname(self):
        return f'{self.nettype}_{self.width_mult:.2f}'


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epoch)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(self.epoch*0.7))
        return (
            {"optimizer": optimizer, "lr_scheduler": scheduler},
            #{"optimizer": optimizer},
        )

    def training_step(self, batch, batch_idx):
        x,y = batch
        pred = self.net(x)
        loss = F.cross_entropy(pred, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.net(x)
        loss = F.cross_entropy(pred, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        pred = torch.argmax(pred, dim=1)
        test_acc = torch.sum(y == pred).item() / (y.shape[0] * 1.0)
        self.log('val_acc', test_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def forward(self, x):
        if not self.quantization and hasattr(self,'executor') and len(self.executor)>0 and self.executor[0] is not None:
            return self.executor[0].forward(x)[0]
        else:
            return self.net(x)

    def test_step(self, test_batch, batch_idx):

        x, y = test_batch
        pred = self.forward(x)
        pred = torch.argmax(pred, dim=1)

        test_acc = torch.sum(y == pred).item() / (y.shape[0] * 1.0)
        self.log('test_acc', test_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)


def tell_macs():
    # for nettype, width_mult in [
    #     ('MobileNetV1', 0.5),
    #     ('MobileNetV2', 0.7),
    #     ('MobileNetV3', 1.65),
    #     # ('MobileNetV3_large',1.),
    #     ('ShuffleNetV1', 1.),
    #     ('ShuffleNetV2', 1.),
    #     ('GhostNet', 1.),
    #     ('MobileNetV1_q', 0.5),
    # ]:
    for nettype, width_mult in [
        ('MobileNetV1', 0.31),
        ('MobileNetV2', 0.43),
        ('MobileNetV3', 1.),
        # ('MobileNetV3_large',1.),
        ('ShuffleNetV1', 0.64),
        ('ShuffleNetV2', 0.63),
        ('GhostNet', 0.6),
        ('MobileNetV1_q', 0.31),
    ]:
        print(f'******************** {nettype} ********************')
        w = ModelWrapper(nettype, width_mult)
        w.summary()

if __name__ == '__main__':
    tell_macs()