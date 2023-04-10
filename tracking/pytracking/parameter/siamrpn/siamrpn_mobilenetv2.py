from .siamrpn_alex import _parameters

def parameters():
    return _parameters(nettype='siamrpn_mobilev2_l234_dwxcorr', quant=False)
