# 为量化生成校准数据库
import torch.nn
import torch.optim as optim
from ltr.dataset import Lasot, Got10k, MSCOCOSeq
from ltr.data import processing, sampler, LTRLoader
from ltr.models.qfnet import qfnet_factory
import ltr.models.loss as ltr_losses
import ltr.actors.tracking as actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.models.loss.bbr_loss import GIoULoss_x1y1x2y2


def run(settings):
    settings.fusion_type = 'concat'
    settings.description = 'CALIBRATION'

    settings.batch_size = 32
    settings.num_workers = 1
    settings.multi_gpu = False

    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.stride = 16
    settings.feature_sz = 15

    settings.center_jitter_factor = {'template': 0., 'search': 4.5}
    settings.scale_jitter_factor = {'template': 0., 'search': 0.5}

    settings.search_area_factor = {'template':2., 'search':256*2/127}
    settings.output_sz = {'template': 127, 'search': 256}


    settings.max_gap = 200
    settings.train_samples_per_epoch = 32*32*2
    settings.val_samples_per_epoch = 32
    settings.val_epoch_interval = 1000
    settings.num_epochs = 1

    settings.weight_reg = 1.0
    settings.weight_cls = 1.0

    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='train')
    #trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    coco_train = MSCOCOSeq(settings.env.coco_dir, version='2017')

    # Validation datasets
    #got10k_val = Got10k(settings.env.got10k_dir, split='val')


    # Data transform
    transform_joint = tfm.Transform(tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    data_processing_train = processing.QFNETProcessing(search_area_factor=settings.search_area_factor,
                                                       output_sz=settings.output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='pair',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint)

    # Train sampler and loader
    dataset_train = sampler.STARKSampler([lasot_train, got10k_train], [1, 1],
                                        samples_per_epoch=settings.train_samples_per_epoch, max_gap=settings.max_gap,
                                        num_search_frames=1, num_template_frames=1,
                                        processing=data_processing_train)

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=0)


    # Create network and actor
    class Dummy_Net(torch.nn.Module):
        def __init__(self):
            super(Dummy_Net, self).__init__()
        def fowrard(self,z,x):
            return z
    net = Dummy_Net()


    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)



    objective = {'giou': GIoULoss_x1y1x2y2(), 'l1':torch.nn.L1Loss()}

    loss_weight = {'giou': 2, 'l1':5}

    actor = actors.Create_Calibration(net=net, objective=objective, loss_weight=loss_weight, name='lighttrack')
    # Optimizer
    optimizer = optim.AdamW([
        {'params': actor.net.parameters(), 'lr': 1e-4},
    ], lr=1e-3, weight_decay=0.0001)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    trainer.train(settings.num_epochs, load_latest=False, fail_safe=False)



