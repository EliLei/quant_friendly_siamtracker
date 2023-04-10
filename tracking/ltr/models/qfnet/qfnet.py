import torch
import torch.nn as nn
from ltr.models.qfnet.fusion_method import *
from ltr.models.backbone.ghostnet import ghostnet_q
from ltr.trainers.ltr_trainer import freeze_batchnorm_layers

class Prediction_Head(nn.Module):
    def __init__(self):
        super(Prediction_Head, self).__init__()

class ChostNet_Wrapper(nn.Module):
    def __init__(self, backbone, feat_depth=12):
        super(ChostNet_Wrapper, self).__init__()

        self.features = backbone.features[:feat_depth]

    def forward(self, x):
        x = self.features(x)
        return x



class QFNet(nn.Module):
    def __init__(self, backbone, fusion_method, corner_head, bbox_head, cls_head, traincls=False):
        super(QFNet, self).__init__()

        self.backbone = ChostNet_Wrapper(backbone)
        self.fusion_method = fusion_method
        self.corner_head = corner_head
        self.bbox_head = bbox_head
        self.cls_head = cls_head

        self.traincls = traincls

    def train(self, mode):
        super().train(mode)
        if mode==True and self.traincls:
            print('frozen parameter for cls training')
            freeze_batchnorm_layers(self.backbone)
            freeze_batchnorm_layers(self.fusion_method)
            freeze_batchnorm_layers(self.corner_head)

    def forward(self, z, x):

        z = self.backbone(z)
        x = self.backbone(x)

        zx = self.fusion_method(z,x)
        if self.corner_head is not None:
            tl_map, br_map = self.corner_head.get_score_map(zx)
            if self.cls_head is not None:
                cls = self.cls_head(zx)
                return tl_map, br_map, cls
            return tl_map, br_map

        elif self.bbox_head is not None:
            bb = self.bbox_head(zx)
            if self.cls_head is not None:
                cls = self.cls_head(zx)
                return bb, cls
            return bb




    def get_bb(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)

        zx = self.fusion_method(z, x)
        if self.corner_head is not None:
            bb = self.corner_head(zx)
        elif self.bbox_head is not None:
            bb = self.bbox_head(zx)
        else:
            raise NotImplementedError

        return bb

    def postprocess_bb(self, onnx_outputs):
        if self.corner_head is not None:
            tl_map,br_map = onnx_outputs
            coorx_tl, coory_tl = self.corner_head.soft_argmax(tl_map)
            coorx_br, coory_br = self.corner_head.soft_argmax(br_map)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.corner_head.img_sz
        elif self.bbox_head is not None:
            return onnx_outputs[0]
        else:
            raise NotImplementedError



    def get_cls(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)

        zx = self.fusion_method(z, x)

        cls = self.cls_head(zx)

        return cls


def qfnet_factory(ntype, size=(112,240), n_channels=128, feat_channels=68 ,inner_channels=64, stride=16, backbone_pretrained=None, width_mult=0.6, summary=False):
    param_dict = {
        'concat':(Fusion_Concatenate, 1.36, CornerHead, None, None),
        'add':(Fusion_PixelWiseAdd, 1.5, CornerHead, None, None),
        'corr':(Fusion_Correlation, 1.7, CornerHead, None, None),
        'attn':(Fusion_Attention, 1.7, CornerHead, None, None),
        'concatbb': (Fusion_Concatenate, 1.36, None, BBoxHead, None),
        'addbb': (Fusion_PixelWiseAdd, 1.5, None, BBoxHead, None),
        'corrbb': (Fusion_Correlation, 1.7, None, BBoxHead, None),
        'attnbb': (Fusion_Attention, 1.7, None, BBoxHead, None),
        'attn_144': (Fusion_Attention, 1.7, CornerHead, None, None),
        'ghostattn': (Fusion_GhostAttn, 1.7, CornerHead, None, None),
        'ghostattn1': (Fusion_GhostAttn1, 1.7, CornerHead, None, None),
        'film':(Fusion_FiLM, 1.7, CornerHead, None, None),
        'ghostattn2x':(Fusion_GhostAttn2x,1.7,CornerHead,None,None),
        'ghostattnL': (Fusion_GhostAttn, 3.0, CornerHead, None, None),
    }

    if ntype not in param_dict:
        raise NotImplementedError

    f_type, f_width_mult, corner, bbox, cls = param_dict[ntype]

    backbone = ghostnet_q(num_classes=1000, width_mult=width_mult, pretrained=backbone_pretrained)
    fusion_method = f_type(size=size, width_mult=f_width_mult, n_channels=n_channels, in_channels=feat_channels ,out_channels=inner_channels, stride=stride)
    if corner is not None:
        corner_head = corner(in_channels=inner_channels, feat_sz=size[1]//stride, stride=stride)
    else:
        corner_head = None

    if bbox is not None:
        bbox_head = bbox(in_channels=inner_channels, feat_sz=size[1]//stride, stride=stride)
    else:
        bbox_head = None

    if cls is not None:
        cls_head = cls(in_channels=inner_channels, feat_sz=size[1]//stride, stride=stride)
    else:
        cls_head = None

    # cls_head = ClsHead(in_channels=inner_channels, feat_sz=size[1]//stride)

    net = QFNet(backbone, fusion_method, corner_head, bbox_head, cls_head)

    if summary:
        import torchinfo
        ret_bbz = torchinfo.summary(net.backbone, (1,3,size[0],size[0]), verbose=0, device='cuda')
        ret_bbx = torchinfo.summary(net.backbone, (1,3,size[1],size[1]), verbose=0, device='cuda')
        print(f'fm type: {ntype}')
        ret_fm = torchinfo.summary(net.fusion_method, ((1,feat_channels,size[0]//stride,size[0]//stride),(1,feat_channels,size[1]//stride,size[1]//stride)), verbose=0, device='cuda')

        if net.corner_head is not None:
            ret_corner = torchinfo.summary(net.corner_head, (1,inner_channels,size[1]//stride,size[1]//stride), verbose=0, device='cuda')
        if net.bbox_head is not None:
            ret_bbox = torchinfo.summary(net.bbox_head, (1,inner_channels,size[1]//stride,size[1]//stride), verbose=0, device='cuda')
        if net.cls_head is not None:
            ret_cls = torchinfo.summary(net.cls_head, (1, inner_channels, size[1] // stride, size[1] // stride), verbose=0,
                                       device='cuda')

        ret_total = torchinfo.summary(net, ((1,3,size[0],size[0]),(1,3,size[1],size[1])), verbose=0, device='cuda')

        def print_net(name, ret):
            print(f"{name}: macs {ret.total_mult_adds/1000000}M input {ret.input_size} out {ret.summary_list[-1].output_size}")

        print_net('bbz', ret_bbz)
        print_net('bbx', ret_bbx)
        print_net('fm',ret_fm)
        if net.corner_head is not None:
            print_net('cn',ret_corner)
        if net.bbox_head is not None:
            print_net('bbox',ret_bbox)
        if net.cls_head is not None:
            print_net('cls',ret_cls)
        print_net('total', ret_total)

        pass

    return net


if __name__ == '__main__':

    for ntype in ['concat','add','corr','attn','concatbb','addbb','corrbb','attnbb','ghostattn','film','ghostattn2x']:
        print(ntype)
        net = qfnet_factory(ntype, summary=True)