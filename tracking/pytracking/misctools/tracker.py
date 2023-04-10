import torchinfo
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

def analyse(net, netname, onnxdir, calidir, reportfile, CALIBRATION = None, ERROR_ANALYSE = None,DEVICE = 'cuda',mask=False,d_uint=False,**params):



    #net = qfnet_factory(nettype, (template_size, search_size))
    #ckptdir = os.path.join(env.workspace_dir, 'checkpoints', 'ltr', 'qfnet', 'qf' + nettype)


    QSetting = QuantizationSettingFactory.default_setting()
    QSetting.quantize_activation_setting.calib_algorithm = 'percentile'
    QSetting.quantize_parameter_setting.calib_algorithm = 'minmax'

    onnx_dir = onnxdir
    #onnx_dir = os.path.join(env.workspace_dir, 'onnx', 'qfnet', nettype)
    os.makedirs(onnx_dir, exist_ok=True)
    f_fp32 = os.path.join(onnx_dir, f'{netname}_fp32_analyse.onnx')
    f_int8 = os.path.join(onnx_dir, f'{netname}_int8_analyse.native')

    if CALIBRATION is None or ERROR_ANALYSE is None:

        dataset = []
        calibration_dir = calidir

        #calibration_dir = os.path.join(env.workspace_dir, 'calibration', 'qfnet')
        z_dir = os.path.join(calibration_dir, 'z')
        x_dir = os.path.join(calibration_dir, 'x')
        if mask:
            zmask_dir = os.path.join(calibration_dir, 'zmask')
            xmask_dir = os.path.join(calibration_dir, 'xmask')
        i = 0
        zs = []
        xs = []
        if mask:
            zmasks = []
            xmasks = []
        while True:
            path_z = os.path.join(z_dir, f'{i:06}.data')
            path_x = os.path.join(x_dir, f'{i:06}.data')
            if mask:
                path_zmask = os.path.join(zmask_dir, f'{i:06}.data')
                path_xmask = os.path.join(xmask_dir, f'{i:06}.data')
            if not os.path.exists(path_z) or not os.path.exists(path_x):
                break
            if mask:
                if not os.path.exists(path_zmask) or not os.path.exists(path_xmask):
                    break
            zs.append(torch.load(path_z))
            xs.append(torch.load(path_x))
            if mask:
                zmasks.append(torch.load(path_zmask))
                xmasks.append(torch.load(path_xmask))
            i += 1
        zs = torch.concat(zs, dim=0)
        xs = torch.concat(xs, dim=0)
        if d_uint:
            zs = zs*255
            xs =  xs*255
        if mask:
            zmasks = torch.concat(zmasks, dim=0).to(torch.float)
            xmasks = torch.concat(xmasks, dim=0).to(torch.float)
        if mask:
            datalist = list(zip(zs.split(1, dim=0), xs.split(1, dim=0), zmasks.split(1, dim=0), xmasks.split(1, dim=0)))
            for z, x, zmask, xmask in datalist:
                dataset.append({'z': z.to(DEVICE), 'x': x.to(DEVICE),
                                'zmask': zmask.to(DEVICE), 'xmask': xmask.to(DEVICE)})
        else:
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
                          input_names=['z', 'x', 'zmask','xmask'] if mask else ['z', 'x'],
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

    #reportfile = os.path.join(onnx_dir, f'{netname}_fp32_analyse.csv')
    report = statistical_analyse(
        graph=quantized, running_device=DEVICE,
        collate_fn=collate_fn, dataloader=ERROR_ANALYSE)

    report = DataFrame(report)
    report.to_csv(reportfile)
    return CALIBRATION, ERROR_ANALYSE

def get_net_params(netname):
    env = env_settings()
    params = {}

    if netname == 'ghostattn1':
        net = qfnet_factory(netname, (112, 240))
        nettype = netname
        ckptdir = os.path.join(env.workspace_dir, 'checkpoints', 'ltr', 'qfnet', 'qf' + nettype)
        ckpts = os.listdir(ckptdir)
        ckpts.sort()
        ckpt = os.path.join(ckptdir, ckpts[-1])
        net.load_state_dict(torch.load(ckpt, map_location='cpu')['net'], strict=True)
        onnxdir = os.path.join(env.workspace_dir, 'onnx', 'qfnet', nettype)
        calidir = os.path.join(env.workspace_dir, 'calibration', 'qfnet')
        netname = f'qfnet_{netname}'
        params['size_z'] = 112
        params['size_x'] = 240
    elif netname == 'lighttrack':
        from pytracking.parameter.lighttrack.default import _parameters
        net = _parameters().net

        class Wrapper(torch.nn.Module):
            def __init__(self, net):
                super(Wrapper, self).__init__()
                self.net = net

            def forward(self, z, x):
                return self.net.trackzx(z, x)

        net = Wrapper(net)
        net = net.to('cuda')

        onnxdir = os.path.join(env.workspace_dir, 'onnx', 'lighttrack')
        calidir = os.path.join(env.workspace_dir, 'calibration', 'lighttrack')
        params['size_z'] = 127
        params['size_x'] = 256
    elif netname == 'starklightning':
        from pytracking.parameter.starklightning.default import _parameters
        net = _parameters().net
        class Wrapper(torch.nn.Module):
            def __init__(self, net):
                super(Wrapper, self).__init__()
                self.net = net

            #def forward(self,z,x):
            def forward(self, z, x, zmask, xmask):
                zdict = self.net.forward_backbone(z, zx="template0", mask=zmask)
                xdict = self.net.forward_backbone(x, zx="search", mask=xmask)

                #zdict = self.net.forward_backbone(z, zx="template0", mask=None)
                #xdict = self.net.forward_backbone(x, zx="search", mask=None)

                # {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}
                q = xdict["feat"]+xdict["pos"]
                k = torch.cat([zdict['feat'],xdict["feat"]],dim=1)+torch.cat([zdict['pos'],xdict["pos"]],dim=1)
                v = torch.cat([zdict['feat'],xdict["feat"]],dim=1)
                key_padding_mask = torch.cat([zdict['mask'],xdict["mask"]],dim=1).unsqueeze(-1)
                key_padding_mask = 1-key_padding_mask
                k=k*key_padding_mask
                v=v*key_padding_mask
                # run the transformer
                out_dict, _, _ = self.net.forward_transformer(q=q, k=k, v=v)
                #out_dict, _, _ = self.net.forward_transformer(q=q, k=k, v=v, key_padding_mask=key_padding_mask)
                pred = out_dict['pred_boxes']

                return pred

        net = Wrapper(net)
        net = net.to('cuda')
        onnxdir = os.path.join(env.workspace_dir, 'onnx', 'starklightning')
        calidir = os.path.join(env.workspace_dir, 'calibration', 'starklightning')
        params['mask'] = True
        params['size_z'] = 128
        params['size_x'] = 320
    elif netname == 'siamrpn_mobilev2_l234_dwxcorr':


        from pytracking.parameter.siamrpn.siamrpn_mobilenetv2 import _parameters
        p= _parameters(netname)
        net = p.net
        cfg = p.cfg


        class Wrapper(torch.nn.Module):
            def __init__(self, net):
                super(Wrapper, self).__init__()
                self.net = net

            def forward(self, z, x):

                zf = self.net.backbone(z)
                if cfg.MASK.MASK:
                    zf = zf[-1]
                if cfg.ADJUST.ADJUST:
                    zf = self.net.neck(zf)
                zf = zf

                xf = self.net.backbone(x)
                if cfg.MASK.MASK:
                    self.xf = xf[:-1]
                    xf = xf[-1]
                if cfg.ADJUST.ADJUST:
                    xf = self.net.neck(xf)
                cls, loc = self.net.rpn_head(zf, xf)
                if cfg.MASK.MASK:
                    mask, self.net.mask_corr_feature = self.mask_head(zf, xf)
                    return cls, loc, mask

                return cls, loc
        net = Wrapper(net).to('cuda')

        onnxdir = os.path.join(env.workspace_dir, 'onnx', 'siamrpn')
        calidir = os.path.join(env.workspace_dir, 'calibration', 'siamrpn_mobilenet')
        params['size_z'] = 127
        params['size_x'] = 255
        params['d_uint'] = True
    elif netname == 'siamrpn_alex_dwxcorr':
        from pytracking.parameter.siamrpn.siamrpn_mobilenetv2 import _parameters
        p = _parameters(netname)
        net = p.net
        cfg = p.cfg

        class Wrapper(torch.nn.Module):
            def __init__(self, net):
                super(Wrapper, self).__init__()
                self.net = net

            def forward(self, z, x):

                zf = self.net.backbone(z)
                if cfg.MASK.MASK:
                    zf = zf[-1]
                if cfg.ADJUST.ADJUST:
                    zf = self.net.neck(zf)
                zf = zf

                xf = self.net.backbone(x)
                if cfg.MASK.MASK:
                    self.xf = xf[:-1]
                    xf = xf[-1]
                if cfg.ADJUST.ADJUST:
                    xf = self.net.neck(xf)
                cls, loc = self.net.rpn_head(zf, xf)
                if cfg.MASK.MASK:
                    mask, self.net.mask_corr_feature = self.mask_head(zf, xf)
                    return cls, loc, mask

                return cls, loc

        net = Wrapper(net).to('cuda')
        onnxdir = os.path.join(env.workspace_dir, 'onnx', 'siamrpn')
        calidir = os.path.join(env.workspace_dir, 'calibration', 'siamrpn')
        params['size_z'] = 127
        params['size_x'] = 287
        params['d_uint'] = True
    else:
        raise NotImplementedError

    reportfile = os.path.join(onnxdir, f'{netname}_fp32_analyse.csv')
    params['net']=net
    params['netname'] = netname
    params['onnxdir'] = onnxdir
    params['calidir'] = calidir
    params['reportfile'] = reportfile
    return params




def do():
    params=[]

    net2name = {'ghostattn1':'Ours','lighttrack':'LightTrack','starklightning':'STARK-Lightning','siamrpn_mobilev2_l234_dwxcorr':'SiamRPN MobileNetV2','siamrpn_alex_dwxcorr':'SiamRPN AlexNet'}

    params.append(['ghostattn1', (294,343,354,),
                   {144,145,154,166,175,195,212,224,233,242,251,
                    270,287,*range(294,300),301,305,306,307,
                    *range(333,355)} ])
    params.append(['lighttrack', (523,537,586,),
                   {262,264,280,281,283,284,286,297,298,317,318,
                      320,321,323,334,335,354,373,392,393,395,396,
                      398,409,410,429,448,467,468,470,471,473,484,
                      485,504,523,525,*range(528,531),536,537,
                      *range(559,587)}])
    params.append(['starklightning', (307,464,508,),
                   {197,199,200,207,209,210,220,221,228,230,231,241,
                    242,252,253,263,264,271,273,274,284,285,295,296,
                    306,307,*range(335,339),431,433,440,441,442,448,
                    450,457,458,459,464,495,500,501,506,507,508,}])
    params.append(['siamrpn_mobilev2_l234_dwxcorr', (328,409,445,),
                   {200,203,204,207,208,211,212,213,216,217,220,221,
                    231,232,235,236,239,240,250,260,264,265,268,269,
                    279,289,299,300,303,304,307,308,318,328,368,405,
                    406,407,*range(409,414),443,444,445}])
    params.append(['siamrpn_alex_dwxcorr', (21,39,43,),
                   {*range(11,22),*range(35,38),*range(39,44)}])

    plot_scale = (0.4,0.7,1.0,100.)

    results = {}

    #fig = plt.figure()
    fig,ax = plt.subplots(figsize=(9,3))

    env = env_settings()
    CALIBRATION, ERROR_ANALYSE = None, None

    for nettype,key_layers, mainnodes in params:

        params = get_net_params(nettype)
        params['CALIBRATION'] = None
        params['ERROR_ANALYSE'] = None
        reportfile = params['reportfile']
        if not os.path.exists(reportfile):
            CALIBRATION, ERROR_ANALYSE = analyse(**params)

        net = params['net']
        net.eval()
        if 'mask' in  params and params['mask']:
            inputshape = ((1,3,params['size_z'],params['size_z']),(1,3,params['size_x'],params['size_x']),(1,params['size_z'],params['size_z']),(1,params['size_x'],params['size_x']))
        else:
            inputshape = ((1,3,params['size_z'],params['size_z']),(1,3,params['size_x'],params['size_x']))
        info = torchinfo.summary(net, inputshape, verbose=0, device='cuda')
        print(f"{nettype}: macs {info.total_mult_adds / 1000000}M params {info.total_params / 1000000}M, "
              f"trainable_params {info.trainable_params / 1000000}M, input_size {info.input_size}")

        df = pandas.read_csv(reportfile)

        ys = [[] for i in range(len(plot_scale))]
        ynames = [[] for i in range(len(plot_scale))]

        current_index = 0

        snr = []

        for index, row in df.iterrows():
            if row['Is output']:
                node_index = int(row['Op name'].split('_')[-1])
                #if node_index in mainnodes or (current_index<len(key_layers) and row['Op name']==key_layers[current_index]):
                #    ys[current_index].append(row['Noise:Signal Power Ratio'])
                if node_index in mainnodes:
                    ys[current_index].append(row['Noise:Signal Power Ratio'])
                #ynames[current_index].append(row['Op name'])

                if current_index<len(key_layers) and node_index>=key_layers[current_index]:
                    current_index+=1
                    snr.append(row['Noise:Signal Power Ratio'])
                # if row['Variable name']=='output0':
                #     snr.append(row['Noise:Signal Power Ratio'])
        print(nettype,' '.join([f'{v:.4f}' for v in snr]))

        # def straight(data):
        #     ret =[]
        #     for d in data:
        #         ret.extend(d)
        #     return ret
        # print('\n'.join([str((y,yn)) for y,yn in zip(straight(ys),straight(ynames))]))


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

    for landmark in plot_scale:
        plt.axvline(landmark, color="lightgrey", linestyle="--")


    plt.xticks([0.]+[((plot_scale[i-1] if i>0 else 0) + plot_scale[i])/2 for i in range(len(plot_scale)-1)]+[1.],['input','backbone','fusion','head','output',])

    #plt.xticks((0.0,1.0),('$f_{x}$','$f_{zx}$'))

    plt.xlim(0.,1.0)
    plt.ylim(0.,4.)
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