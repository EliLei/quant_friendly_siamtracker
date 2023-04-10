from ltr.models.pysot.models.model_builder import ModelBuilder
from pytracking.utils import TrackerParams
from ltr.models.qfnet import qfnet_factory
import torch
import os
from ltr.admin.environment import env_settings
from ltr.models.pysot.utils.model_load import load_pretrain
from ltr.models.pysot.core.config import cfg


def parameters():
    return _parameters(quant=False)

def _parameters(nettype='siamrpn_alex_dwxcorr', quant=False):
    params = TrackerParams()

    params.quant=quant

    params.device = 'cuda'


    env = env_settings()
    snapshot = os.path.join(env.pretrained_networks, nettype, 'model.pth')
    cfgfile = os.path.join(env.pretrained_networks, nettype, 'config.yaml')

    cfg.merge_from_file(args.config)

    # create model
    model = ModelBuilder()
    # load model
    model = load_pretrain(model, snapshot)
    params.net = model






    return params
