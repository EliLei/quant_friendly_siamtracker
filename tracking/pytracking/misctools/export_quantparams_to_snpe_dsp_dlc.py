import os


from ppq.utils.write_qparams_to_snpe_dlc import write_qparams_to_dlc_model, json_load

from ltr.admin.environment import env_settings


nettype='ghostattn1'

env = env_settings()

onnx_dir = os.path.join(env.workspace_dir, 'onnx', 'qfnet', nettype)
f_dlc = os.path.join(onnx_dir, f'qfnet_{nettype}_fakeint8.dlc')
f_config = os.path.join(onnx_dir, f'qfnet_{nettype}_ppq.table')
f_q_dlc = os.path.join(onnx_dir, f'qfnet_{nettype}_int8.dlc')

act_ranges = json_load(f_config)['activation_encodings']
write_qparams_to_dlc_model(f_dlc, f_q_dlc, act_ranges)