from matplotlib import pyplot as plt
import os
import torch
from matplotlib.ticker import MultipleLocator
from ppq import QuantizationSettingFactory, TargetPlatform, TorchExecutor, graphwise_error_analyse, \
    layerwise_error_analyse, statistical_analyse
from ppq.api import quantize_onnx_model, load_native_graph, export_ppq_graph
from torchvision import transforms
import numpy as np
import time
import onnx
from pandas import DataFrame
import pandas
from ltr.admin.environment import env_settings
from ltr.models.qfnet import qfnet_factory

def analyse(nettype, template_size=112, search_size=240, CALIBRATION = None, ERROR_ANALYSE = None,DEVICE = 'cuda'):

    env = env_settings()

    net = qfnet_factory(nettype, (template_size, search_size))
    ckptdir = os.path.join(env.workspace_dir, 'checkpoints', 'ltr', 'qfnet', 'qf' + nettype)
    ckpts = os.listdir(ckptdir)
    ckpts.sort()
    ckpt = os.path.join(ckptdir, ckpts[-1])
    net.load_state_dict(torch.load(ckpt, map_location='cpu')['net'], strict=True)

    QSetting = QuantizationSettingFactory.default_setting()
    QSetting.quantize_activation_setting.calib_algorithm = 'percentile'
    QSetting.quantize_parameter_setting.calib_algorithm = 'minmax'

    onnx_dir = os.path.join(env.workspace_dir, 'onnx', 'qfnet', nettype)
    os.makedirs(onnx_dir, exist_ok=True)
    f_fp32 = os.path.join(onnx_dir, f'qfnet_{nettype}_fp32_analyse.onnx')
    f_int8 = os.path.join(onnx_dir, f'qfnet_{nettype}_int8_analyse.native')

    if CALIBRATION is None or ERROR_ANALYSE is None:

        dataset = []

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
            dataset.append({'z': z.to(DEVICE), 'x': x.to(DEVICE)})
        CALIBRATION = dataset[:len(dataset) // 2]
        ERROR_ANALYSE = dataset[len(dataset) // 2:]

        print(f"Calibration dataset loaded: len {len(CALIBRATION)}")
        print(f"Error analyse dataset loaded: len {len(ERROR_ANALYSE)}")

    datasample = CALIBRATION[0]
    BATCH_SIZE = 32

    def collate_fn(batch: dict) -> torch.Tensor:
        return {k: v.to(DEVICE) for k, v in batch.items()}

    if os.path.exists(f_int8):
        quantized = load_native_graph(f_int8)
        print(f'load quant model from {f_int8}')
    else:
        net = net.to(DEVICE)
        net.eval()



        # 这里 pytorch版本为 1.13， 导出时会有些参数作为indentity的输出，传递给conv，ppq无法处理conv动态参数，
        # 需要使用低版本 如 1.10
        torch.onnx.export(net, (datasample,),
                          f_fp32,
                          input_names=['z', 'x'],
                          #output_names=['x1y1x2y2'],
                          # dynamic_axes={"z": [0], "x": [0], "tl_map": [0], "br_map": [0]},
                          opset_version=13,
                          keep_initializers_as_inputs=False,
                          )
        quantized = quantize_onnx_model(f_fp32, CALIBRATION,
                                        calib_steps=len(CALIBRATION) // BATCH_SIZE,
                                        input_shape=None, platform=TargetPlatform.SNPE_INT8,
                                        setting=QSetting, collate_fn=collate_fn,
                                        inputs=datasample, device=DEVICE)

        export_ppq_graph(graph=quantized, platform=TargetPlatform.NATIVE,
                         graph_save_to=f_int8)
        print(f'save quant model to {f_int8}')

    reportfile = os.path.join(onnx_dir, f'qfnet_{nettype}_fp32_analyse.csv')
    report = statistical_analyse(
        graph=quantized, running_device=DEVICE,
        collate_fn=collate_fn, dataloader=ERROR_ANALYSE)

    report = DataFrame(report)
    report.to_csv(reportfile)
    return CALIBRATION, ERROR_ANALYSE



def do():
    params=[]

    net2name = {'concat':'Concatenate', 'corr':'DW-Correlation','attn':'Attention','film':'FiLM', 'ghostattn1':'GhostAttention'}

    params.append(['concat', (294, 321),
                   {*range(294, 300), 303,304,*range(313, 322)} ])
    params.append(['corr', (294, 521),
                   {*range(294, 315), *range(513, 522)}])
    params.append(['attn', (294, 343),
                   {*range(294, 300), 301,*range(305, 309),311,312,317,318,323, *range(325, 344)}])
    params.append(['film', (294, 314),
                   {*range(294, 300), *range(305, 315)}])
    params.append(['ghostattn1', (294, 343),
                   {*range(294, 300), 301, *range(305, 309),311,312,317,318,323,*range(325, 333), *range(333, 344)}])

    plot_scale = (0.,1.,100)

    results = {}

    #fig = plt.figure()
    fig,ax = plt.subplots(figsize=(9,3))

    env = env_settings()
    CALIBRATION, ERROR_ANALYSE = None, None

    for nettype,key_layers, mainnodes in params:

        onnx_dir = os.path.join(env.workspace_dir, 'onnx', 'qfnet', nettype)
        os.makedirs(onnx_dir, exist_ok=True)
        reportfile = os.path.join(onnx_dir, f'qfnet_{nettype}_fp32_analyse.csv')

        if not os.path.exists(reportfile):
            CALIBRATION, ERROR_ANALYSE = analyse(nettype,CALIBRATION=CALIBRATION, ERROR_ANALYSE=ERROR_ANALYSE)
        df = pandas.read_csv(reportfile)

        ys = [[] for i in range(len(plot_scale))]

        current_index = 0

        snr = []

        for index, row in df.iterrows():
            if row['Is output']:
                node_index = int(row['Op name'].split('_')[-1])
                #if node_index in mainnodes or (current_index<len(key_layers) and row['Op name']==key_layers[current_index]):
                #    ys[current_index].append(row['Noise:Signal Power Ratio'])
                if node_index in mainnodes:
                    ys[current_index].append(row['Noise:Signal Power Ratio'])

                if current_index<len(key_layers) and node_index>=key_layers[current_index]:
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

        plt.plot(xs,ys,label=net2name[nettype] if nettype in net2name else nettype)
        #print(xs.tolist(),ys.tolist())


        pass

    plt.axvline(0.0, color="lightgrey", linestyle="--")
    plt.axvline(1.0, color="lightgrey", linestyle="--")


    #plt.xticks([((plot_scale[i-1] if i>0 else 0) + plot_scale[i])/2 for i in range(len(plot_scale)-1)]+[plot_scale[-1]],['stride=2','stride=4','stride=8','stride=16','stride=32','output',])

    plt.xticks((0.0,1.0),('$f_{x}$','$f_{zx}$'))

    plt.xlim(0.,1.0)
    plt.ylim(0.,0.5)
    #plt.ylim(0.0000001,100)
    plt.xlabel('')
    plt.ylabel(r'$SNR=\left(\frac{noise}{signal}\right)^{2}$')
    #plt.yscale('log')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    plt.subplots_adjust(left=0.075, bottom=0.15, right=0.97, top=0.925,)

    plt.legend()
    plt.show()




if __name__ == '__main__':
    do()