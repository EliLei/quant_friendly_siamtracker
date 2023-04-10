# https://github.com/megvii-model/ShuffleNet-Series
import torch
import torch.nn as nn

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        # batchsize, num_channels, height, width = x.data.size()
        # assert (num_channels % 4 == 0)
        # x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        # x = x.permute(1, 0, 2)
        # x = x.reshape(2, -1, num_channels // 2, height, width)
        # return x[0], x[1]
        num_channels = x.shape[1]
        return x[:,:num_channels//2,...], x[:,num_channels//2:,...]

class ShuffleNetV2(nn.Module):
    def __init__(self, input_size=224, n_class=1000, width_mult=1.0):
        super(ShuffleNetV2, self).__init__()
        self.width_mult = width_mult

        self.stage_repeats = [4, 8, 4]
        group = 4
        self.stage_out_channels = [-1,
                                   int(24*width_mult)//group*group,
                                   int(116*width_mult)//group*group,
                                   int(232*width_mult)//group*group,
                                   int(464*width_mult)//group*group,
                                   int(1024*width_mult)//group*group]
        #    [-1, 24, 116, 232, 464, 1024]

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
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(input_channel, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=1))

                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.ReLU(inplace=True)
        )
        self.globalpool = nn.AvgPool2d(7)
        if self.width_mult >= 2.0:
            self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.globalpool(x)
        if self.width_mult >= 2.0:
            x = self.dropout(x)
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
    model = ShuffleNetV2()
    # print(model)

    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())

    import os
    import onnxruntime
    import numpy as np

    f = os.path.join(os.path.split(__file__)[0], f'shufflenet_v2.onnx')
    torch.onnx.export(model, test_data, f, input_names=('input0',),output_names=('output0',),opset_version=11,dynamic_axes={'input0':[0],'output0':[0]})
    onnx_model = onnxruntime.InferenceSession(f, providers=['CUDAExecutionProvider'])
    data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    onnx_input = {onnx_model.get_inputs()[0].name: data}
    outputs = onnx_model.run(None, onnx_input)

    pass
