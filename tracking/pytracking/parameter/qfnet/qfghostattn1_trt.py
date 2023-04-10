from ppq import TargetPlatform

from .qfconcat import _parameters

def parameters():
    return _parameters('ghostattn1', True,platform=TargetPlatform.TRT_INT8)
