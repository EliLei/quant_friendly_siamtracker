# https://github.com/zhaoyuzhi/PyTorch-MobileNet-v123
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def weights_init(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """


    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    print('Initialize network with %s type' % init_type)
    net.apply(init_func)


###========================== MobileNetv3 framework ==========================
def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        #y = self.avg_pool(x).view(b, c)
        # view() 会使导出的onnx固定batch大小
        y = self.avg_pool(x)
        y = y.squeeze(-1).squeeze(-1)
        # y = self.fc(y).view(b, c, 1, 1)
        y = self.fc(y)
        y = y.unsqueeze(-1).unsqueeze(-1)
        # return x * y.expand_as(x)
        # expand_as导致导出的onnx难以推断数据的shape
        return x * y


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MobileBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, kernel, stride, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and in_channels == out_channels

        if nl == 'RE':
            nonlinear_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nonlinear_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, latent_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(latent_channels),
            nonlinear_layer(inplace=True),
            # dw
            nn.Conv2d(latent_channels, latent_channels, kernel, stride, padding, groups=latent_channels, bias=False),
            nn.BatchNorm2d(latent_channels),
            SELayer(latent_channels),
            nonlinear_layer(inplace=True),
            # pw-linear
            nn.Conv2d(latent_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


'''
if mode == 'large':
    # refer to Table 1 in paper
    mobile_setting = [
        # k, exp, c,  se,     nl,  s,
        [3, 16,  16,  False, 'RE', 1],
        [3, 64,  24,  False, 'RE', 2],
        [3, 72,  24,  False, 'RE', 1],
        [5, 72,  40,  True,  'RE', 2],
        [5, 120, 40,  True,  'RE', 1],
        [5, 120, 40,  True,  'RE', 1],
        [3, 240, 80,  False, 'HS', 2],
        [3, 200, 80,  False, 'HS', 1],
        [3, 184, 80,  False, 'HS', 1],
        [3, 184, 80,  False, 'HS', 1],
        [3, 480, 112, True,  'HS', 1],
        [3, 672, 112, True,  'HS', 1],
        [5, 672, 160, True,  'HS', 2],
        [5, 960, 160, True,  'HS', 1],
        [5, 960, 160, True,  'HS', 1],
    ]
elif mode == 'small':
    # refer to Table 2 in paper
    mobile_setting = [
        # k, exp, c,  se,     nl,  s,
        [3, 16,  16,  True,  'RE', 2],
        [3, 72,  24,  False, 'RE', 2],
        [3, 88,  24,  False, 'RE', 1],
        [5, 96,  40,  True,  'HS', 2],
        [5, 240, 40,  True,  'HS', 1],
        [5, 240, 40,  True,  'HS', 1],
        [5, 120, 48,  True,  'HS', 1],
        [5, 144, 48,  True,  'HS', 1], 
        [5, 288, 96,  True,  'HS', 2],
        [5, 576, 96,  True,  'HS', 1],
        [5, 576, 96,  True,  'HS', 1],
    ]
else:
    raise NotImplementedError
'''


class MobileNetV3_large(nn.Module):
    def __init__(self, n_class=1000, dropout=0.8, width_mult=1.0):
        super(MobileNetV3_large, self).__init__()
        # Start Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(width_mult*16), 3, 2, 1, bias=False),
            nn.BatchNorm2d(int(width_mult*16)),
            Hswish(inplace=True)
        )
        # MobileBottleneck blocks
        self.conv2 = MobileBottleneck(int(width_mult*16), int(width_mult*16), int(width_mult*16), 3, 1, False, 'RE')
        self.conv3 = MobileBottleneck(int(width_mult*16), int(width_mult*24), int(width_mult*64), 3, 2, False, 'RE')
        self.conv4 = MobileBottleneck(int(width_mult*24), int(width_mult*24), int(width_mult*72), 3, 1, False, 'RE')
        self.conv5 = MobileBottleneck(int(width_mult*24), int(width_mult*40), int(width_mult*72), 5, 2, True, 'RE')
        self.conv6 = MobileBottleneck(int(width_mult*40), int(width_mult*40), int(width_mult*120), 5, 1, True, 'RE')
        self.conv7 = MobileBottleneck(int(width_mult*40), int(width_mult*40), int(width_mult*120), 5, 1, True, 'RE')
        self.conv8 = MobileBottleneck(int(width_mult*40), int(width_mult*80), int(width_mult*240), 3, 2, False, 'HS')
        self.conv9 = MobileBottleneck(int(width_mult*80), int(width_mult*80), int(width_mult*200), 3, 1, False, 'HS')
        self.conv10 = MobileBottleneck(int(width_mult*80), int(width_mult*80), int(width_mult*184), 3, 1, False, 'HS')
        self.conv11 = MobileBottleneck(int(width_mult*80), int(width_mult*80), int(width_mult*184), 3, 1, False, 'HS')
        self.conv12 = MobileBottleneck(int(width_mult*80), int(width_mult*112), int(width_mult*480), 3, 1, True, 'HS')
        self.conv13 = MobileBottleneck(int(width_mult*112), int(width_mult*112), int(width_mult*672), 3, 1, True, 'HS')
        self.conv14 = MobileBottleneck(int(width_mult*112), int(width_mult*160), int(width_mult*672), 5, 2, True, 'HS')
        self.conv15 = MobileBottleneck(int(width_mult*160), int(width_mult*160), int(width_mult*960), 5, 1, True, 'HS')
        self.conv16 = MobileBottleneck(int(width_mult*160), int(width_mult*160), int(width_mult*960), 5, 1, True, 'HS')
        # Last Conv
        self.conv17 = nn.Sequential(
            nn.Conv2d(int(width_mult*160), int(width_mult*960), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(width_mult*960)),
            Hswish(inplace=True)
        )
        self.conv18 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int(width_mult*960), int(width_mult*1280), 1, 1, 0),
            Hswish(inplace=True)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(int(width_mult*1280), n_class)
        )

    def forward(self, x):
        # feature extraction
        x = self.conv1(x)  # out: B * 16 * 112 * 112
        x = self.conv2(x)  # out: B * 16 * 112 * 112
        x = self.conv3(x)  # out: B * 24 * 56 * 56
        x = self.conv4(x)  # out: B * 24 * 56 * 56
        x = self.conv5(x)  # out: B * 40 * 28 * 28
        x = self.conv6(x)  # out: B * 40 * 28 * 28
        x = self.conv7(x)  # out: B * 40 * 28 * 28
        x = self.conv8(x)  # out: B * 80 * 14 * 14
        x = self.conv9(x)  # out: B * 80 * 14 * 14
        x = self.conv10(x)  # out: B * 80 * 14 * 14
        x = self.conv11(x)  # out: B * 80 * 14 * 14
        x = self.conv12(x)  # out: B * 112 * 14 * 14
        x = self.conv13(x)  # out: B * 112 * 14 * 14
        x = self.conv14(x)  # out: B * 160 * 7 * 7
        x = self.conv15(x)  # out: B * 160 * 7 * 7
        x = self.conv16(x)  # out: B * 160 * 7 * 7
        x = self.conv17(x)  # out: B * 960 * 7 * 7
        x = self.conv18(x)  # out: B * 1280 * 1 * 1
        # classifier
        x = x.mean(3).mean(2)  # out: B * 1280 (global avg pooling)
        x = self.classifier(x)  # out: B * 1000
        return x


class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, dropout=0.8, width_mult=1.0):
        super(MobileNetV3, self).__init__()
        # Start Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(width_mult*16), 3, 2, 1, bias=False),
            nn.BatchNorm2d(int(width_mult*16)),
            Hswish(inplace=True)
        )
        # MobileBottleneck blocks
        self.conv2 = MobileBottleneck(int(width_mult*16), int(width_mult*16), int(width_mult*16), 3, 2, True, 'RE')
        self.conv3 = MobileBottleneck(int(width_mult*16), int(width_mult*24), int(width_mult*72), 3, 2, False, 'RE')
        self.conv4 = MobileBottleneck(int(width_mult*24), int(width_mult*24), int(width_mult*88), 3, 1, False, 'RE')
        self.conv5 = MobileBottleneck(int(width_mult*24), int(width_mult*40), int(width_mult*96), 5, 2, True, 'HS')
        self.conv6 = MobileBottleneck(int(width_mult*40), int(width_mult*40), int(width_mult*240), 5, 1, True, 'HS')
        self.conv7 = MobileBottleneck(int(width_mult*40), int(width_mult*40), int(width_mult*240), 5, 1, True, 'HS')
        self.conv8 = MobileBottleneck(int(width_mult*40), int(width_mult*48), int(width_mult*120), 5, 1, True, 'HS')
        self.conv9 = MobileBottleneck(int(width_mult*48), int(width_mult*48), int(width_mult*144), 5, 1, True, 'HS')
        self.conv10 = MobileBottleneck(int(width_mult*48), int(width_mult*96), int(width_mult*288), 5, 2, True, 'HS')
        self.conv11 = MobileBottleneck(int(width_mult*96), int(width_mult*96), int(width_mult*576), 3, 1, True, 'HS')
        self.conv12 = MobileBottleneck(int(width_mult*96), int(width_mult*96), int(width_mult*576), 3, 1, True, 'HS')
        # Last Conv
        self.conv13 = nn.Sequential(
            nn.Conv2d(int(width_mult*96), int(width_mult*576), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(width_mult*576)),
            Hswish(inplace=True)
        )
        self.conv14 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int(width_mult*576), int(width_mult*1280), 1, 1, 0),
            Hswish(inplace=True)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(int(width_mult*1280), n_class)
        )

    def forward(self, x):
        # feature extraction
        x = self.conv1(x)  # out: B * 16 * 112 * 112
        x = self.conv2(x)  # out: B * 16 * 56 * 56
        x = self.conv3(x)  # out: B * 24 * 28 * 28
        x = self.conv4(x)  # out: B * 24 * 28 * 28
        x = self.conv5(x)  # out: B * 40 * 14 * 14
        x = self.conv6(x)  # out: B * 40 * 14 * 14
        x = self.conv7(x)  # out: B * 40 * 14 * 14
        x = self.conv8(x)  # out: B * 48 * 14 * 14
        x = self.conv9(x)  # out: B * 48 * 14 * 14
        x = self.conv10(x)  # out: B * 96 * 7 * 7
        x = self.conv11(x)  # out: B * 96 * 7 * 7
        x = self.conv12(x)  # out: B * 96 * 7 * 7
        x = self.conv13(x)  # out: B * 576 * 7 * 7
        x = self.conv14(x)  # out: B * 1280 * 1 * 1
        # classifier
        x = x.mean(3).mean(2)  # out: B * 1280 (global avg pooling)
        x = self.classifier(x)  # out: B * 1000
        return x


if __name__ == "__main__":
    # net = MobileNetV3_large()
    net = MobileNetV3(last_channels=1280, n_class=1000, dropout=0)
    weights_init(net, init_type='normal', init_gain=0.02)
    a = torch.randn(1, 3, 224, 224)
    b = net(a)
    print(b)
    print(b.shape)