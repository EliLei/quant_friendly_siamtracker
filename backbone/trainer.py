import os

import pytorch_lightning as pl

from argparse import ArgumentParser

from lightning_wrapper import ImageNetDataModule, ModelWrapper



def getlogdir():
    ret = os.path.join(os.path.split(__file__)[0], 'logs')
    if not os.path.exists(ret):
        os.makedirs(ret)
    return ret

def find_last_ckpt(logdir):
    if not os.path.exists(logdir):
        return None
    lightningdir = os.path.join(logdir,'lightning_logs')
    if not os.path.exists(lightningdir):
        return None
    versiondirs = os.listdir(lightningdir)
    versiondirs.sort(key=lambda x: int(x.split('_')[1]))
    if len(versiondirs)<1:
        return None
    versiondir = os.path.join(lightningdir, versiondirs[-1])
    ckptdir = os.path.join(versiondir, 'checkpoints')
    # ckpts = os.listdir(ckptdir)
    # ckpts = [ckpt for ckpt in ckpts if os.path.splitext(ckpt)[1]=='.ckpt']
    # ckpts.sort(key=lambda x: int(x.split('-')[0].split('=')[1]))

    # if len(ckpts)<1:
    #     return None
    # return os.path.join(ckptdir,ckpts[-1])
    ckpt =os.path.join(ckptdir,'last.ckpt')
    if os.path.exists(ckpt):
        return ckpt
    return None

if __name__ == '__main__':
    #tell_macs()


    #cmdline = r"--model=MobileNetV1 --width_mult=0.5 --accelerator=gpu --devices=1 --max_epochs=60 --batch_size=512"


    parser = ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--width_mult", type=float, default=1.)

    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default='1')
    parser.add_argument("--max_epochs", type=int, default=50)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--load_last", action='store_true')

    #args = parser.parse_args(cmdline.split())
    args = parser.parse_args()

    model = ModelWrapper(nettype=args.model, width_mult=args.width_mult, epoch=args.max_epochs, )
    datamodule = ImageNetDataModule(batch_size=args.batch_size, num_workers=args.num_workers)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=True, filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}')
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, max_epochs=args.max_epochs,
                         default_root_dir=os.path.join(getlogdir(), model.netname()),
                         auto_scale_batch_size=True, callbacks=[checkpoint_callback])


    if args.load_last:
        ckpt = find_last_ckpt(os.path.join(getlogdir(), model.netname()))
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt)
    else:
        trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)
    model.quantize(device=next(model.parameters()).device.__str__())
    model.quantization=True
    trainer.test(model, datamodule=datamodule)


