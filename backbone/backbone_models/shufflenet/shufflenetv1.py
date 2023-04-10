# https://github.com/megvii-model/ShuffleNet-Series
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShuffleV1Block(nn.Module):
    def __init__(self, inp, oup, *, group, first_group, mid_channels, ksize, stride):
        super(ShuffleV1Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.group = group

        if stride == 2:
            outputs = oup - inp
        else:
            outputs = oup

        branch_main_1 = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, groups=1 if first_group else group, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        ]


        branch_main_2 = [
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(outputs),
        ]
        self.branch_main_1 = nn.Sequential(*branch_main_1)
        self.branch_main_2 = nn.Sequential(*branch_main_2)

        # self.channel_shuffle = nn.ChannelShuffle(self.group)
        # self.group_channels = mid_channels//mid_channels
        if self.group > 1:
            self.shuffle_kernel = self.create_channel_shuffle_conv_kernel(mid_channels, self.group)

        if stride == 2:
            self.branch_proj = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, old_x):
        x = old_x
        x_proj = old_x
        x = self.branch_main_1(x)
        if self.group > 1:
            x = self.channel_shuffle(x)
        x = self.branch_main_2(x)
        if self.stride == 1:
            return F.relu(x + x_proj)
        elif self.stride == 2:
            return torch.cat((self.branch_proj(x_proj), F.relu(x)), 1)

    # 卷积代替 channel_shuffle https://zhuanlan.zhihu.com/p/203549964 原文代码有误
    def create_channel_shuffle_conv_kernel(self, num_channels, num_groups):
        channels_per_group = num_channels // num_groups
        conv_kernel = torch.zeros(num_channels, num_channels, 1, 1)
        for k in range(num_channels):
            index = (k % channels_per_group) * num_groups + k // channels_per_group
            conv_kernel[k, index, 0, 0] = 1
        return conv_kernel

    def channel_shuffle(self, x):
        """用卷积实现 channel shuffle
        """
        device = x.device
        if self.shuffle_kernel.device != device:
            self.shuffle_kernel = self.shuffle_kernel.to(device)
        return torch.conv2d(x, self.shuffle_kernel)

    # def channel_shuffle(self, x):
    #     batchsize, num_channels, height, width = x.data.size()
    #     #assert num_channels % self.group == 0
    #     group_channels = num_channels // self.group
    #
    #     x = x.reshape(batchsize, group_channels, self.group, height, width)
    #     x = x.permute(0, 2, 1, 3, 4)
    #     x = x.reshape(batchsize, num_channels, height, width)
    #
    #     return x




class ShuffleNetV1(nn.Module):
    def __init__(self, n_class=1000, group=3,width_mult=1.0):
        super(ShuffleNetV1, self).__init__()
        #print('model size is ', model_size)

        assert group is not None

        self.stage_repeats = [4, 8, 4]
        if group == 3:
            self.stage_out_channels = [-1, int(24*width_mult)//group*group, int(240*width_mult)//group*group, int(480*width_mult)//group*group, int(960*width_mult)//group*group]
        elif group == 8:

            self.stage_out_channels = [-1, int(24*width_mult)//group*group, int(384*width_mult)//group*group, int(768*width_mult)//group*group, int(1536*width_mult)//group*group]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                first_group = idxstage == 0 and i == 0
                self.features.append(ShuffleV1Block(input_channel, output_channel,
                                            group=group, first_group=first_group,
                                            mid_channels=(output_channel // 4)//group*group, ksize=3, stride=stride))
                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        self.globalpool = nn.AvgPool2d(7)

        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)

        x = self.globalpool(x)
        #x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        # (B C1) * (C1 C2) 会被ppq当成matmul，不量化？强行转换成带bias的fc层
        if not self.training and self.classifier[0].bias is None:
            self.classifier[0].bias = nn.Parameter(torch.zeros(self.classifier[0].out_features,device=x.device,dtype=x.dtype))
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    model = ShuffleNetV1(group=3).cuda()
    # print(model)

    test_data = torch.rand(5, 3, 224, 224).cuda()
    test_outputs = model(test_data)
    print(test_outputs.size())