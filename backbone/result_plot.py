from matplotlib import pyplot as plt
import os
import torch
from matplotlib.ticker import MultipleLocator
from ppq import QuantizationSettingFactory, TargetPlatform, TorchExecutor, graphwise_error_analyse, \
    layerwise_error_analyse, statistical_analyse
from ppq.api import quantize_onnx_model
from torchvision import transforms
from imagenet_dataset import ImageNet
import numpy as np
import time
import onnx
from pandas import DataFrame
import pandas

CALIBRATION = {}
ERROR_ANALYSE = {}

def analyse(nettype, width_mult):

    f_fp32 = os.path.join(os.path.split(__file__)[0], 'logs', f'{nettype}_{width_mult:.2f}_fp32.onnx')

    device = 'cuda'
    input_size = (3,224,224)


    #data = torch.randn(1, *self.input_size)
    # torch.onnx.export(self.net, data, f_fp32,
    #                   input_names=('input0',),
    #                   output_names=('output0',),
    #                   opset_version=13,
    #                   dynamic_axes={'input0': [0], 'output0': [0]},
    #                   keep_initializers_as_inputs=False,
    #                   )

    def load_calibration_dataset(error_analyse=False):
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
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
        np.random.seed(int(time.time() * 1000) & 0xffffffff)

        return [torch.stack([dataset[post_quantization_idx[i + j * 32]][0] for i in range(32)]) for j in range(32)]
        # return [torch.randn(INPUT_SHAPE) for j in range(32)]

    def collate_fn(batch: torch.Tensor) -> torch.Tensor:
        return batch.to(device)

    if input_size not in CALIBRATION:
        CALIBRATION[input_size] = load_calibration_dataset()


    if input_size not in ERROR_ANALYSE:
        ERROR_ANALYSE[input_size] = load_calibration_dataset(error_analyse=True)

    QSetting = QuantizationSettingFactory.default_setting()
    QSetting.quantize_activation_setting.calib_algorithm = 'percentile'
    QSetting.quantize_parameter_setting.calib_algorithm = 'minmax'

    quantized = quantize_onnx_model(f_fp32, CALIBRATION[input_size], calib_steps=32,
                                    input_shape=(1, *input_size), collate_fn=collate_fn,
                                    setting=QSetting, platform=TargetPlatform.SNPE_INT8, device=device, verbose=0)

    # graphwise = graphwise_error_analyse(graph=quantized, running_device=device, collate_fn=collate_fn,
    #                         dataloader=ERROR_ANALYSE[input_size])

    # layerwise = layerwise_error_analyse(
    #     graph=quantized, running_device=device, collate_fn=collate_fn,
    #     dataloader=ERROR_ANALYSE[input_size])

    report = statistical_analyse(
        graph=quantized, running_device=device,
        collate_fn=collate_fn, dataloader=ERROR_ANALYSE[input_size])


    reportfile = os.path.join(os.path.split(__file__)[0], 'logs', f'{nettype}_{width_mult:.2f}.csv')



    report = DataFrame(report)
    report.to_csv(reportfile)

    pass
    # export_ppq_graph(graph=quantized, platform=TargetPlatform.NATIVE,
    #                  graph_save_to=f_int8)
    # export_ppq_graph(graph=quantized, platform=TargetPlatform.SNPE_INT8,
    #                  graph_save_to=os.path.splitext(f_int8)[0])



def show_result():
    pass

def do():
    params=[]

    net2name = {'MobileNetV1':'MobileNet v1 ReLU','MobileNetV2':'MobileNet v2','MobileNetV3':'MobileNet v3','MobileNetV1Relu6':'MobileNet v1 ReLU6',
                'ShuffleNetV1':'ShuffleNet v1', 'ShuffleNetV2':'ShuffleNet v2','GhostNet':'GhostNet'}

    params.append(['MobileNetV1', 0.31, ('Relu_5', 'Relu_13', 'Relu_21', 'Relu_45','Relu_53'),
                   {*range(0, 57)}])
    params.append(['MobileNetV1Relu6', 0.31, ('Clip_11', 'Clip_27', 'Clip_43', 'Clip_91', 'Clip_107'),
                   {*range(0, 111)}])
    params.append(['MobileNetV2', 0.43, ('Clip_12', 'Clip_31', 'Clip_60', 'Clip_128', 'Clip_166'),
                   {*range(0, 18), *range(27, 37),46,*range(56, 66),75,85,*range(95, 105),114,*range(124, 134),143,*range(153, 170)}])
    params.append(['MobileNetV3', 1.00, ('Relu_9', 'Relu_33', 'Div_50', 'Div_233', 'Div_343'),
                   {0, *range(5, 11),*range(29, 37),*range(42, 44),*range(48, 52),70,*range(75, 79),115,*range(152, 154),*range(158, 162),180,*range(185, 189),
                    *range(225, 227),*range(231, 235),253,*range(258, 262),298,*range(335, 337),*range(341, 346),*range(350, 356)}])
    params.append(['ShuffleNetV1', 0.64, ('Relu_1', 'Relu_4', 'Relu_37', 'Relu_104', 'Relu_139'),
                   {*range(0, 3),13,20,21,28,29,36,37,48,55,56,63,64,71,72,
                    79,80,87,88,95,96,103,104,115,122,123,130,131,*range(138, 148)}])
    params.append(['ShuffleNetV2', 0.63, ('Relu_1', 'Relu_7', 'Concat_59', 'Concat_180', 'Relu_239'),
                   {*range(0, 3),11,27,43,59,68,84,100,116,132,148,164,180,189,205,221,*range(237, 248),}])
    params.append(['GhostNet', 0.60, ('Add_20', 'Add_61', 'Add_130', 'Add_277', 'Relu_419'),
                   {*range(0, 3),20,42,61,97,130,152,171,190,209,244,277,313,332,365,384,*range(417, 432)}])

    plot_scale = (0.1,0.3,0.5,0.7,0.9,1.)

    results = {}

    #fig = plt.figure()
    fig,ax = plt.subplots(figsize=(9,3))

    for nettype, width_mult,key_layers, mainnodes in params:
        reportfile = os.path.join(os.path.split(__file__)[0], 'logs', f'{nettype}_{width_mult:.2f}.csv')
        if not os.path.exists(reportfile):
            analyse(nettype, width_mult)
        df = pandas.read_csv(reportfile)

        ys = [[] for i in range(len(plot_scale))]

        current_index = 0

        snr = []

        for index, row in df.iterrows():
            if row['Is output']:
                node_index = int(row['Op name'].split('_')[-1])
                if node_index in mainnodes or (current_index<len(key_layers) and row['Op name']==key_layers[current_index]):
                    ys[current_index].append(row['Noise:Signal Power Ratio'])

                if current_index<len(key_layers) and row['Op name']==key_layers[current_index]:
                    current_index+=1
                    snr.append(row['Noise:Signal Power Ratio'])
                if row['Variable name']=='output0':
                    snr.append(row['Noise:Signal Power Ratio'])
        print(nettype,' '.join([f'{v:.4f}' for v in snr]))


        xs = []
        start_x=0.
        for current_index in range(len(ys)):
            end_x = plot_scale[current_index]
            xs.append(np.linspace(start_x, end_x,num=len(ys[current_index])+1)[1:])

            start_x=end_x
        xs = np.concatenate([np.array([0.])]+xs)
        ys = np.concatenate([np.array([0.])]+[np.array(yline) for yline in ys ])

        plt.plot(xs,ys,label=net2name[nettype])
        print(xs.tolist(),ys.tolist())


        pass

    plt.axvline(plot_scale[0], color="lightgrey", linestyle="--")
    plt.axvline(plot_scale[1], color="lightgrey", linestyle="--")
    plt.axvline(plot_scale[2], color="lightgrey", linestyle="--")
    plt.axvline(plot_scale[3], color="lightgrey", linestyle="--")
    plt.axvline(plot_scale[4], color="lightgrey", linestyle="--")
    plt.axvline(plot_scale[5], color="lightgrey", linestyle="--")




    plt.xticks([((plot_scale[i-1] if i>0 else 0) + plot_scale[i])/2 for i in range(len(plot_scale)-1)]+[plot_scale[-1]],['stride=2','stride=4','stride=8','stride=16','stride=32','output',])

    plt.xlim(0.,1.)
    plt.ylim(0.,2.)
    #plt.ylim(0.0000001,100)
    plt.xlabel('')
    plt.ylabel(r'$SNR=\left(\frac{noise}{signal}\right)^{2}$')
    #plt.yscale('log')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    plt.subplots_adjust(left=0.075, bottom=0.15, right=0.97, top=0.925,)

    plt.legend()
    plt.show()




if __name__ == '__main__':
    do()