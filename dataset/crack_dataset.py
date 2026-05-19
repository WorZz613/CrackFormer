import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
from dataset import transforms as T

class train_transform:
    def __init__(self,size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = []
        trans.extend([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class eval_transform:
    def __init__(self,size,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)



class Crack_Dataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(Crack_Dataset, self).__init__()

        self.flag = "train" if train else "val"
        self.transforms = transforms

        self.img_path = []

        images_dataset = os.listdir(os.path.join(root,'images',self.flag))
        images_dataset = [images_dataset[i] for i in range(len(images_dataset)) if len(images_dataset[i]) > 2]
        for im in images_dataset:
            im_fp = os.path.join(root,'images',self.flag,im)
            lb_fp = os.path.join(root, 'masks', self.flag,im.replace('.jpg','.png'))
            self.img_path.append({
                'image': im_fp,
                'mask': lb_fp
            })
        print(f'{len(self.img_path)} samples for {self.flag}')
        pass
    def __getitem__(self, idx):

        img = Image.open(self.img_path[idx]['image']).convert('RGB')
        mask = Image.open(self.img_path[idx]['mask']).convert('L')
        mask = np.array(mask)
        mask[mask > 0] = 1
        mask = Image.fromarray(mask,mode = 'L')
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask

    def __len__(self):
        return len(self.img_path)
