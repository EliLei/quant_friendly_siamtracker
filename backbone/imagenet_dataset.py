import torch
import torchvision
from torch.utils.data import Dataset
import local
import os
from PIL import Image

class ImageNet(Dataset):
    def __init__(self, split='train', transform=None):
        self.IMAGENET_INFO = {}
        self.IMAGENET_TAG2IDX = {}
        with open('imagenet_mapping.txt') as f:
            lines = f.readlines()
        for l in lines:
            l = l.strip().split()
            self.IMAGENET_INFO[int(l[0])] = l[1:]
            self.IMAGENET_TAG2IDX[l[1]] = int(l[0])

        self.transform = transform

        self.split = split
        self.data = []

        if split=='train':
            #bbox_root = os.path.join(local.IMAGENET_DIR,'ILSVRC2012_bbox_train_v2')
            img_root = os.path.join(local.IMAGENET_DIR,'ILSVRC2012_img_train')
        elif split=='val':
            img_root = os.path.join(local.IMAGENET_DIR, 'ILSVRC2012_img_val')
        else:
            raise NotImplementedError

        subdirs = os.listdir(img_root)
        for subdir in subdirs:
            anno = self.IMAGENET_TAG2IDX[subdir]
            subdir = os.path.join(img_root, subdir)
            for img in os.listdir(subdir):
                self.data.append((os.path.join(subdir, img), anno))


    def __getitem__(self, item):
        img, anno = self.data[item]
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img,anno

    def __len__(self):
        return self.data.__len__()