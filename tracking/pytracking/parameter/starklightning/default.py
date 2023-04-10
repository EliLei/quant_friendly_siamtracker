from ltr.models.lighttrack.models.models import LightTrackM_Subnet
from ltr.models.stark import build_stark_lightning_x_trt
from pytracking.utils import TrackerParams
from ltr.models.qfnet import qfnet_factory
import torch
import os
from ltr.admin.environment import env_settings



def parameters():
    return _parameters(quant=False)

def _parameters(quant=False):
    params = TrackerParams()

    params.quant=quant

    params.device = 'cuda'


    env = env_settings()
    ckpt = os.path.join(env.pretrained_networks, 'STARKLightningXtrt_net.ckpt')
    cfg = r"{'MODEL': {'HEAD_TYPE': 'CORNER_LITE_REP_v2', 'HIDDEN_DIM': 128, 'HEAD_DIM': 128, 'BACKBONE': {'TYPE': 'RepVGG-A0', 'OUTPUT_LAYERS': ['stage3'], 'DILATION': False, 'LAST_STAGE_BLOCK': 4}, 'TRANSFORMER': {'NHEADS': 8, 'DROPOUT': 0.1, 'DIM_FEEDFORWARD': 1024}}, 'TRAIN': {'DISTILL': False, 'DISTILL_LOSS_TYPE': 'KL', 'AMP': False, 'LR': 0.0001, 'WEIGHT_DECAY': 0.0001, 'EPOCH': 500, 'LR_DROP_EPOCH': 400, 'BATCH_SIZE': 16, 'NUM_WORKER': 8, 'OPTIMIZER': 'ADAMW', 'BACKBONE_MULTIPLIER': 0.1, 'GIOU_WEIGHT': 2.0, 'L1_WEIGHT': 5.0, 'DEEP_SUPERVISION': False, 'FREEZE_BACKBONE_BN': True, 'BACKBONE_TRAINED_LAYERS': ['stage1', 'stage2', 'stage3'], 'PRINT_INTERVAL': 50, 'VAL_EPOCH_INTERVAL': 20, 'GRAD_CLIP_NORM': 0.1, 'SCHEDULER': {'TYPE': 'step', 'DECAY_RATE': 0.1}}, 'DATA': {'MEAN': [0.485, 0.456, 0.406], 'STD': [0.229, 0.224, 0.225], 'MAX_SAMPLE_INTERVAL': 200, 'TRAIN': {'DATASETS_NAME': ['LASOT', 'GOT10K_vottrain', 'COCO17'], 'DATASETS_RATIO': [1, 1, 1], 'SAMPLE_PER_EPOCH': 60000}, 'VAL': {'DATASETS_NAME': ['GOT10K_votval'], 'DATASETS_RATIO': [1], 'SAMPLE_PER_EPOCH': 10000}, 'SEARCH': {'SIZE': 320, 'FEAT_SIZE': 20, 'FACTOR': 5.0, 'CENTER_JITTER': 4.5, 'SCALE_JITTER': 0.5}, 'TEMPLATE': {'SIZE': 128, 'FEAT_SIZE': 8, 'FACTOR': 2.0, 'CENTER_JITTER': 0, 'SCALE_JITTER': 0}}, 'TEST': {'TEMPLATE_FACTOR': 2.0, 'TEMPLATE_SIZE': 128, 'SEARCH_FACTOR': 5.0, 'SEARCH_SIZE': 320, 'EPOCH': 500}, 'ckpt_dir': '/data/users/leirulin/code/Tracking/cache_tracker'}"
    cfg = cfg.replace(r"'",r'"')
    cfg = cfg.replace("True","true")
    cfg = cfg.replace("False", "false")
    from easydict import EasyDict as edict
    from json import loads
    cfg = loads(cfg)
    cfg = edict(cfg)
    params.cfg=cfg
    net = build_stark_lightning_x_trt(cfg, 'test')

    statedict = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(statedict)
    params.net = net


    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    # whether to save boxes from all queries
    params.save_all_boxes = False


    return params
