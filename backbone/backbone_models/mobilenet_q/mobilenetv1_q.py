# https://github.com/zhaoyuzhi/PyTorch-MobileNet-v123
import torch
import torch.nn as nn
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


###========================== MobileNetv1 framework ==========================
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DWConv, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            # nn.BatchNorm2d(in_channels),
            # nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MobileNetV1_q(nn.Module):
    def __init__(self, n_class=1000, width_mult=1.0):
        super(MobileNetV1_q, self).__init__()

        # Start Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(width_mult*32), 3, 2, 1, bias=False),
            nn.BatchNorm2d(int(width_mult*32)),
            nn.ReLU(inplace=True)
        )
        # DWConv blocks
        self.conv2 = DWConv(int(width_mult*32), int(width_mult*64), 1)
        self.conv3 = DWConv(int(width_mult*64), int(width_mult*128), 2)
        self.conv4 = DWConv(int(width_mult*128), int(width_mult*128), 1)
        self.conv5 = DWConv(int(width_mult*128), int(width_mult*256), 2)
        self.conv6 = DWConv(int(width_mult*256), int(width_mult*256), 1)
        self.conv7 = DWConv(int(width_mult*256), int(width_mult*512), 2)
        self.conv8 = DWConv(int(width_mult*512), int(width_mult*512), 1)
        self.conv9 = DWConv(int(width_mult*512), int(width_mult*512), 1)
        self.conv10 = DWConv(int(width_mult*512), int(width_mult*512), 1)
        self.conv11 = DWConv(int(width_mult*512), int(width_mult*512), 1)
        self.conv12 = DWConv(int(width_mult*512), int(width_mult*512), 1)
        self.conv13 = DWConv(int(width_mult*512), int(width_mult*1024), 2)
        self.conv14 = DWConv(int(width_mult*1024), int(width_mult*1024), 1)
        # Classifier
        self.classifier = nn.Linear(int(width_mult*1024), n_class)

    def forward(self, x):
        # feature extraction
        x = self.conv1(x)  # out: B * 32 * 112 * 112
        x = self.conv2(x)  # out: B * 64 * 112 * 112
        x = self.conv3(x)  # out: B * 128 * 56 * 56
        x = self.conv4(x)  # out: B * 128 * 56 * 56
        x = self.conv5(x)  # out: B * 256 * 28 * 28
        x = self.conv6(x)  # out: B * 256 * 28 * 28
        x = self.conv7(x)  # out: B * 512 * 14 * 14
        x = self.conv8(x)  # out: B * 512 * 14 * 14
        x = self.conv9(x)  # out: B * 512 * 14 * 14
        x = self.conv10(x)  # out: B * 512 * 14 * 14
        x = self.conv11(x)  # out: B * 512 * 14 * 14
        x = self.conv12(x)  # out: B * 512 * 14 * 14
        x = self.conv13(x)  # out: B * 1024 * 7 * 7
        x = self.conv14(x)  # out: B * 1024 * 7 * 7
        # classifier
        x = x.mean(3).mean(2)  # out: B * 1024 (global avg pooling)
        x = self.classifier(x)  # out: B * 1000
        return x


if __name__ == "__main__":
    net = MobileNetV1()
    a = torch.randn(1, 3, 224, 224)
    b = net(a)
    print(b.shape)