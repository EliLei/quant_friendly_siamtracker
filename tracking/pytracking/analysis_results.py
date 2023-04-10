
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

trackers.extend(trackerlist(name='qfnet', parameter_name='qfconcat',
                            run_ids=None, display_name='concat'))
trackers.extend(trackerlist(name='qfnet', parameter_name='qfconcat_q',
                            run_ids=None, display_name='concat_q'))
trackers.extend(trackerlist(name='qfnet', parameter_name='qfadd',
                            run_ids=None, display_name='add'))
trackers.extend(trackerlist(name='qfnet', parameter_name='qfadd_q',
                            run_ids=None, display_name='add_q'))
trackers.extend(trackerlist(name='qfnet', parameter_name='qfcorr',
                            run_ids=None, display_name='corr'))
trackers.extend(trackerlist(name='qfnet', parameter_name='qfcorr_q',
                            run_ids=None, display_name='corr_q'))
trackers.extend(trackerlist(name='qfnet', parameter_name='qfattn',
                            run_ids=None, display_name='attn'))
trackers.extend(trackerlist(name='qfnet', parameter_name='qfattn_q',
                            run_ids=None, display_name='attn_q'))
trackers.extend(trackerlist(name='qfnet', parameter_name='qfattnbb',
                            run_ids=None, display_name='attnbb'))
trackers.extend(trackerlist(name='qfnet', parameter_name='qfattnbb_q',
                            run_ids=None, display_name='attnbb_q'))
trackers.extend(trackerlist(name='lighttrack', parameter_name='default',
                            run_ids=None, display_name='lighttrack'))
trackers.extend(trackerlist(name='lighttrack', parameter_name='quant',
                            run_ids=None, display_name='lighttrack_q'))
trackers.extend(trackerlist(name='starklightning', parameter_name='default',
                            run_ids=None, display_name='starklightning'))
trackers.extend(trackerlist(name='starklightning', parameter_name='quant',
                            run_ids=None, display_name='starklightning_q'))
trackers.extend(trackerlist(name='siamrpn', parameter_name='siamrpn_alex',
                            run_ids=None, display_name='siamrpn_alex'))
trackers.extend(trackerlist(name='siamrpn', parameter_name='siamrpn_alex_q',
                            run_ids=None, display_name='siamrpn_alex_q'))
trackers.extend(trackerlist(name='siamrpn', parameter_name='siamrpn_mobilenetv2',
                            run_ids=None, display_name='siamrpn_mobilenetv2'))
trackers.extend(trackerlist(name='siamrpn', parameter_name='siamrpn_mobilenetv2_q',
                            run_ids=None, display_name='siamrpn_mobilenetv2_q'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfattn_144',
#                             run_ids=None, display_name='attn_144'))
# trackers.extend(trackerlist(name='qfnet', parameter_name='qfattn_144_q',
#                             run_ids=None, display_name='attn_144_q'))
trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn',
                            run_ids=None, display_name='qfghostattn'))
trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn_q',
                            run_ids=None, display_name='qfghostattn_q'))
trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1',
                            run_ids=None, display_name='qfghostattn1'))
trackers.extend(trackerlist(name='qfnet', parameter_name='qfghostattn1_q',
                            run_ids=None, display_name='qfghostattn1_q'))


dataset = get_dataset(dataset_name)
plot_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'prec'),
              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
#
#
print_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))


#print_per_sequence_results(trackers,dataset,dataset_name,merge_results=True)
