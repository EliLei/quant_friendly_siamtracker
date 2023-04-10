from ltr.models.lighttrack.models.models import LightTrackM_Subnet
from pytracking.utils import TrackerParams
from ltr.models.qfnet import qfnet_factory
import torch
import os
from ltr.admin.environment import env_settings
from easydict import EasyDict as edict

def parameters():
    params = TrackerParams()

    siam_info = edict()
    siam_info.arch = 'LightTrackM_Subnet'
    siam_info.dataset = 'VOT2019'
    siam_info.epoch_test = False
    siam_info.stride = 16
    # build tracker

    params.info = siam_info

    env = env_settings()
    ckpt = os.path.join(env.pretrained_networks, 'LightTrackM.pth')

    siam_net = LightTrackM_Subnet(path_name='back_04502514044521042540+cls_211000022+reg_100000111_ops_32',stride=16)
    siam_net = load_pretrain(siam_net, ckpt)

    siam_net.eval()

    params.net = siam_net

    return params

def remove_prefix(state_dict, prefix):
    '''
    Old style model is stored with all names of parameters share common prefix 'module.'
    '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def check_keys(model, pretrained_state_dict, print_unuse=True):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = list(ckpt_keys - model_keys)
    missing_keys = list(model_keys - ckpt_keys)

    # remove num_batches_tracked
    for k in sorted(missing_keys):
        if 'num_batches_tracked' in k:
            missing_keys.remove(k)

    print('missing keys:{}'.format(missing_keys))
    if print_unuse:
        print('unused checkpoint keys:{}'.format(unused_pretrained_keys))
    # print('used keys:{}'.format(used_pretrained_keys))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def load_pretrain(model, pretrained_path, print_unuse=True):
    print('load pretrained model from {}'.format(pretrained_path))

    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    # print(pretrained_dict.keys())
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        # print(pretrained_dict.keys())
        pretrained_dict = remove_prefix(pretrained_dict, 'feature_extractor.')  # remove online train
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')  # remove multi-gpu label
        pretrained_dict = remove_prefix(pretrained_dict, 'feature_extractor.')  # remove online train

    check_keys(model, pretrained_dict, print_unuse=print_unuse)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
