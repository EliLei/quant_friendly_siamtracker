
import os
from re import M
import torch
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
#matplotlib.rcParams['font.sans-serif'] = ['SimSun']

from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'lasot'
#dataset_name = 'lasot'
"""stark"""

# trackers.extend(trackerlist(name='qfnet', parameter_name='qfconcat',
#                             run_ids=None, display_name='concat'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfconcat_q',
#                             run_ids=None, display_name='concat_q'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfadd',
#                             run_ids=None, display_name='add'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfadd_q',
#                             run_ids=None, display_name='add_q'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfcorr',
#                             run_ids=None, display_name='corr'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfcorr_q',
#                             run_ids=None, display_name='corr_q'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfattn',
#                             run_ids=None, display_name='attn'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfattn_q',
#                             run_ids=None, display_name='attn_q'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfattnbb',
#                             run_ids=None, display_name='attnbb'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfattnbb_q',
#                             run_ids=None, display_name='attnbb_q'))
# trackers.extend(trackerlist(name='lighttrack', parameter_name='default',
#                             run_ids=None, display_name='lighttrack'))
# trackers.extend(trackerlist(name='lighttrack', parameter_name='quant',
#                             run_ids=None, display_name='lighttrack_q'))
# trackers.extend(trackerlist(name='starklightning', parameter_name='default',
#                             run_ids=None, display_name='starklightning'))
# trackers.extend(trackerlist(name='starklightning', parameter_name='quant',
#                             run_ids=None, display_name='starklightning_q'))
# trackers.extend(trackerlist(name='siamrpn', parameter_name='siamrpn_alex',
#                             run_ids=None, display_name='siamrpn_alex'))
# trackers.extend(trackerlist(name='siamrpn', parameter_name='siamrpn_alex_q',
#                             run_ids=None, display_name='siamrpn_alex_q'))
# trackers.extend(trackerlist(name='siamrpn', parameter_name='siamrpn_mobilenetv2',
#                             run_ids=None, display_name='siamrpn_mobilenetv2'))
# trackers.extend(trackerlist(name='siamrpn', parameter_name='siamrpn_mobilenetv2_q',
#                             run_ids=None, display_name='siamrpn_mobilenetv2_q'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfattn_144',
#                             run_ids=None, display_name='attn_144'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfattn_144_q',
#                             run_ids=None, display_name='attn_144_q'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn',
#                             run_ids=None, display_name='qfghostattn'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn_q',
#                             run_ids=None, display_name='qfghostattn_q'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1',
#                             run_ids=None, display_name='qfghostattn1'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1_q',
#                             run_ids=None, display_name='qfghostattn1_q'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qffilm',
#                             run_ids=None, display_name='qffilm'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qffilm_q',
#                             run_ids=None, display_name='qffilm_q'))



# trackers.extend(trackerlist(name='qfnet', parameter_name='qfconcat',
#                             run_ids=None, display_name='Concatenate'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfcorr',
#                             run_ids=None, display_name='DW-Correlation'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfattn',
#                             run_ids=None, display_name='Attention'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1',
#                             run_ids=None, display_name='GhostAttention'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qffilm',
#                             run_ids=None, display_name='FiLM'))

#
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfconcat_q',
#                             run_ids=None, display_name='concat_q'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfcorr_q',
#                             run_ids=None, display_name='corr_q'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfattn_q',
#                             run_ids=None, display_name='attn_q'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1_q',
#                             run_ids=None, display_name='qfghostattn1_q'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qffilm_q',
#                             run_ids=None, display_name='qffilm_q'))

trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1_q',
                            run_ids=None, display_name='Ours'))
trackers.extend(trackerlist(name='lighttrack', parameter_name='quant',
                            run_ids=None, display_name='LightTrack'))
trackers.extend(trackerlist(name='starklightning', parameter_name='quant',
                            run_ids=None, display_name='STARK-Lightning'))
trackers.extend(trackerlist(name='siamrpn', parameter_name='siamrpn_alex_q',
                            run_ids=None, display_name='SiamRPN AlexNet'))
trackers.extend(trackerlist(name='siamrpn', parameter_name='siamrpn_mobilenetv2_q',
                            run_ids=None, display_name='SiamRPN MobileNetV2'))
#
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1',
#                             run_ids=None, display_name='Ours-unq'))
# trackers.extend(trackerlist(name='lighttrack', parameter_name='default',
#                             run_ids=None, display_name='LightTrack-unq'))
# trackers.extend(trackerlist(name='starklightning', parameter_name='default',
#                             run_ids=None, display_name='STARK-Lightning-unq'))
# trackers.extend(trackerlist(name='siamrpn', parameter_name='siamrpn_alex',
#                             run_ids=None, display_name='SiamRPN AlexNet-unq'))
# trackers.extend(trackerlist(name='siamrpn', parameter_name='siamrpn_mobilenetv2',
#                             run_ids=None, display_name='SiamRPN MobileNetV2-unq'))

# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1',
#                             run_ids=None, display_name='Float32'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1_q',
#                             run_ids=None, display_name='SNPE'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1_ncnn',
#                             run_ids=None, display_name='NCNN'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1_openvino',
#                             run_ids=None, display_name='OpenVINO'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1_qnn',
#                             run_ids=None, display_name='QNN'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1_nxp',
#                             run_ids=None, display_name='NXP'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1_trt',
#                             run_ids=None, display_name='TensorRT'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1_tengine',
#                             run_ids=None, display_name='Tengine'))


dataset = get_dataset(dataset_name)
plot_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'prec'),
              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

print_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))


# from matplotlib import pyplot as plt
# points = (('Ours',0.52,110,45.58),('STARK-Lightning',2.28,1726,0.09),('LightTrack',1.79,594,3.11),('SiamRPN MobileNetv2',11.15,7837,43.84),('SiamRPN Alex',6.25,7178,43.08))
#
# scalar=120
# for p in points:
#     name,params,macs,auc = p
#     plt.scatter(macs,auc,s=params*scalar,alpha=0.3, cmap='viridis')
#
# plt.show()
#print_per_sequence_results(trackers,dataset,dataset_name,merge_results=True)
