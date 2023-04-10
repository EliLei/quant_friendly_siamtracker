from .ghostnet import *
from .mobilenet import *
from .shufflenet import *

def backbone_factory(nettype, width_mul=1.):
    if globals().get(nettype)==1:
        return 1

if __name__ == '__main__':
    print(globals().get('ghostnet'))
    pass