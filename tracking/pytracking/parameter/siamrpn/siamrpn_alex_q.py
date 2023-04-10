from .siamrpn_alex import _parameters

def parameters():
    return _parameters(nettype='siamrpn_alex_dwxcorr', quant=True)
