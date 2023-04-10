from ppq import TargetPlatform

from pytracking.utils import TrackerParams
from ltr.models.qfnet import qfnet_factory
import torch
import os
from ltr.admin.environment import env_settings
import threading

def parameters():
    return _parameters('concat', False)

def _parameters(nettype, quant=False, template_size=112, search_size=240, platform=TargetPlatform.SNPE_INT8):
    params = TrackerParams()

    params.target_platform = platform

    params.nettype = nettype
    params.debug = 0
    params.visualization = False

    params.use_gpu = True
    params.quant = quant

    params.template_size = template_size
    params.template_factor = 2.0
    params.search_size = search_size
    params.search_factor = params.template_factor * params.search_size / params.template_size

    params.net = qfnet_factory(nettype, (params.template_size, params.search_size))
    env = env_settings()
    ckptdir = os.path.join(env.workspace_dir,'checkpoints','ltr','qfnet','qf'+nettype)
    ckpts = os.listdir(ckptdir)
    ckpts.sort()
    ckpt = os.path.join(ckptdir, ckpts[-1])
    params.net.load_state_dict(torch.load(ckpt, map_location='cpu')['net'], strict=True)

    # TRT_INT8 = 101
    # NCNN_INT8 = 102
    # OPENVINO_INT8 = 103
    # TENGINE_INT8 = 104
    #
    # PPL_CUDA_INT8 = 201
    # PPL_CUDA_INT4 = 202
    # PPL_CUDA_FP16 = 203
    # PPL_CUDA_MIX = 204
    #
    # PPL_DSP_INT8 = 301
    # SNPE_INT8 = 302
    # PPL_DSP_TI_INT8 = 303
    # QNN_DSP_INT8 = 304
    #
    # HOST_INT8 = 401
    #
    # NXP_INT8 = 501
    # FPGA_INT8 = 502
    #
    # ORT_OOS_INT8 = 601
    #
    # METAX_INT8_C = 701  # channel wise
    # METAX_INT8_T = 702  # tensor wise
    #
    # HEXAGON_INT8 = 801
    #
    # FP32 = 0
    # # SHAPE-OR-INDEX related operation
    # SHAPE_OR_INDEX = -1
    # # initial state
    # UNSPECIFIED = -2
    # # boundary op
    # BOUNDARY = -3
    # # just used for calling exporter
    # ONNX = -4
    # CAFFE = -5
    # NATIVE = -6
    # ONNXRUNTIME = -7
    # # THIS IS A DUUMY PLATFORM JUST FOR CREATING YOUR OWN EXTENSION.
    # EXTENSION = -10086
    #
    # ACADEMIC_INT8 = 10081
    # ACADEMIC_INT4 = 10082
    # ACADEMIC_MIX = 10083



    return params
