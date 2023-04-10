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
    _default_settings(settings)
    settings.fusion_type = 'concat'
    settings.description = 'QFCONCAT'
    _run(settings)

def _default_settings(settings):
    settings.batch_size = 32
    settings.num_workers = 8
    settings.multi_gpu = False

    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.stride = 16
    settings.feature_sz = 15

    settings.center_jitter_factor = {'template': 0., 'search': 4.5}
    settings.scale_jitter_factor = {'template': 0., 'search': 0.5}

    settings.search_area_factor = {'template': 2., 'search': 240 * 2 / 112}
    settings.output_sz = {'template': 112, 'search': 240}

    settings.max_gap = 200
    settings.train_samples_per_epoch = 60000
    settings.val_samples_per_epoch = 10000
    settings.val_epoch_interval = 5
    settings.num_epochs = 90

    settings.weight_reg = 1.0
    settings.weight_cls = 1.0

def _run(settings):

    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='train')
    #trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    coco_train = MSCOCOSeq(settings.env.coco_dir, version='2017')

    # Validation datasets
    got10k_val = Got10k(settings.env.got10k_dir, split='val')


    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip(probability=0.5),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))



    data_processing_train = processing.QFNETProcessing(search_area_factor=settings.search_area_factor,
                                                       output_sz=settings.output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='pair',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint)

    data_processing_val = processing.QFNETProcessing(search_area_factor=settings.search_area_factor,
                                                       output_sz=settings.output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='pair',
                                                       transform=transform_val,
                                                       joint_transform=transform_joint)

    # Train sampler and loader
    dataset_train = sampler.STARKSampler([lasot_train, got10k_train, coco_train], [1, 1, 1],
                                        samples_per_epoch=settings.train_samples_per_epoch, max_gap=settings.max_gap,
                                        num_search_frames=1, num_template_frames=1,
                                        processing=data_processing_train)

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=0)

    # Validation samplers and loaders
    dataset_val = sampler.STARKSampler([got10k_val], [1], samples_per_epoch=settings.val_samples_per_epoch,
                                      max_gap=settings.max_gap,
                                      num_search_frames=1, num_template_frames=1,
                                      processing=data_processing_val)

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=settings.val_epoch_interval, stack_dim=0)

    # Create network and actor
    net = qfnet_factory(settings.fusion_type, (settings.output_sz['template'],settings.output_sz['search']),
                        backbone_pretrained=settings.env.pretrained_networks+'ghostnet_0.6_imagenet1000_epoch60.ckpt',
                        summary=True)


    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)



    objective = {'giou': GIoULoss_x1y1x2y2(), 'l1':torch.nn.L1Loss()}

    loss_weight = {'giou': 2, 'l1':5}

    actor = actors.QFBBActor(net=net, objective=objective, loss_weight=loss_weight)
    # Optimizer
    paramlist = [{'params': actor.net.backbone.parameters(), 'lr': 1e-4},
        {'params': actor.net.fusion_method.parameters(), 'lr': 1e-3},]
    if actor.net.corner_head is not None:
        paramlist.append({'params': actor.net.corner_head.parameters(), 'lr': 1e-3})
    if actor.net.bbox_head is not None:
        paramlist.append({'params': actor.net.bbox_head.parameters(), 'lr': 1e-3})
    optimizer = optim.AdamW(paramlist, lr=1e-3, weight_decay=0.0001)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[settings.num_epochs//3, settings.num_epochs//3*2], gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(settings.num_epochs, load_latest=True, fail_safe=True)



    # dataset_train.train_cls=True
    # dataset_val.train_cls=True
    # objective = {'bce': torch.nn.BCEWithLogitsLoss}
    # loss_weight = {'bce': 1}
    #
    # actor = actors.QFClsActor(net=net, objective=objective, loss_weight=loss_weight)
    # #freeze backbone bn
    # actor.net.traincls=True
    #
    # #重新设置 optimizer
    # trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings)
    # trainer.load_checkpoint()
    # optimizer = optim.AdamW([
    #     {'params': actor.net.cls_head.parameters(), 'lr': 1e-4},
    # ], lr=1e-4, weight_decay=0.0001)
    # trainer.optimizer = optimizer
    # dataset_train.frame_sample_mode = 'trident_pro'
    # dataset_val.frame_sample_mode = 'trident_pro'
    #
    # trainer.train(settings.num_epochs+5, load_latest=False, fail_safe=False)
