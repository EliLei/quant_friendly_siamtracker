# https://github.com/openppl-public/ppq/blob/76e03261bad580e7c52e6f0856034fa9313f69b5/md_doc/inference_with_snpe_dsp.md

import os

import numpy as np
import torch

from ppq import QuantizationSettingFactory
from ppq.api import dispatch_graph, export_ppq_graph, load_onnx_graph, load_native_graph
from ppq.core import TargetPlatform
from ppq.executor import TorchExecutor


from ltr.admin.environment import env_settings


nettype='ghostattn1'

env = env_settings()

onnx_dir = os.path.join(env.workspace_dir, 'onnx', 'qfnet', nettype)
f_int8 = os.path.join(onnx_dir, f'qfnet_{nettype}_int8.native')
f_graph = os.path.join(onnx_dir, f'qfnet_{nettype}_ppq')
f_config = os.path.join(onnx_dir, f'qfnet_{nettype}_ppq.table')

ppq_graph_ir = load_native_graph(f_int8)

# export quantization param file and model file
export_ppq_graph(graph=ppq_graph_ir, platform=TargetPlatform.QNN_DSP_INT8, graph_save_to=f_graph, config_save_to=f_config)


