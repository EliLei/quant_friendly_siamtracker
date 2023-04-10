from .ghostnet import *
from .mobilenet import *
from .shufflenet import *
from .mobilenet_q import *

def backbone_factory(nettype, width_mult=1.):
    classname = globals().get(nettype)
    if classname is not None:
        return classname(width_mult=width_mult)

    return None