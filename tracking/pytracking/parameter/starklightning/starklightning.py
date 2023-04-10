from pytracking.utils import TrackerParams
from ltr.models.qfnet import qfnet_factory
import torch
import os
from ltr.admin.environment import env_settings
import threading

def parameters():
    return _parameters('concat', False)

class MultiThreadLock:
    @classmethod
    def getlock(cls):
        if not hasattr(cls, 'singleton_lock'):
            setattr(cls, 'singleton_lock', threading.Lock())
        return getattr(cls, 'singleton_lock')

def _parameters(nettype, quant=False):
    params = TrackerParams()

    params.nettype = nettype
    params.debug = 0
    params.visualization = False

    params.use_gpu = True
    params.quant = quant

    params.template_size = 112
    params.template_factor = 2.0
    params.search_size = 240
    params.search_factor = params.template_factor * params.search_size / params.template_size

    params.net = qfnet_factory(nettype, (params.template_size, params.search_size))
    env = env_settings()
    ckptdir = os.path.join(env.workspace_dir,'checkpoints','ltr','qfnet','qf'+nettype)
    ckpts = os.listdir(ckptdir)
    ckpts.sort()
    ckpt = os.path.join(ckptdir, ckpts[-1])
    params.net.load_state_dict(torch.load(ckpt, map_location='cpu')['net'], strict=True)


    params.lock = MultiThreadLock.getlock()

    return params
