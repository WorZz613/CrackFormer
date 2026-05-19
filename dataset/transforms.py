from typing import List, Union

import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):

        if target is not None:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target
        else:
            for t in self.transforms:
                image = t(image)
            return image



class Resize(object):
    def __init__(self, size:Union[int,tuple,List]):
        '''
        :param size: (h,w)
        '''
        if isinstance(size,int):
            self.size = [size,size]
        elif isinstance(size,(tuple,list)):
            self.size = size

    def __call__(self, image, target):
        size = self.size
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class Resize_scale(object):
    def __init__(self, scale: int):

        self.scale = scale

    def __call__(self, image, target):
        size = image.size

        new_size = [size[1] // self.scale,size[0] // self.scale]

        image = F.resize(image, new_size)
        target = F.resize(target, new_size, interpolation=T.InterpolationMode.NEAREST)
        return image, target



class Resize_val(object):
    def __init__(self, scale: int):

        self.scale = scale

    def __call__(self, image, target=None):
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        size = image.size
        size = [size[0] // self.scale, size[1] // self.scale]
        new_size = [int(64 * (size[0] // 64)),int(64 * (size[1] // 64))]

        image = F.resize(image, new_size)
        if target is not None:
            target = F.resize(target, new_size, interpolation=T.InterpolationMode.NEAREST)
            return image, target
        else:
            return image


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None):
        image = pad_if_smaller(image, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        if target is not None:
            target = pad_if_smaller(target, self.size, fill = 255)
            target = F.crop(target, *crop_params)
            return image, target
        else:
            return image


class Pad(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None):
        image = F.center_crop(image, self.size)
        if target is not  None:
            target = F.center_crop(target, self.size)
            return image, target
        else:
            return image

class ToTensor(object):
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is not None:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
            return image, target
        else:
            return image


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is not None:
            return image, target
        else:
            return image


#自适应直方图均衡化CLAHE
class CLAHE(object):
    def __init__(self,contrast_factor=2.0):
        self.contrast_factor = contrast_factor

    def __call__(self,image,target=None):
        image = F.adjust_contrast(image,self.contrast_factor)
        if target is not None:
            return image, target
        else:
            return image


class Gamma(object):
    def __init__(self,gamma=0.5):
        self.gamma = gamma

    def __call__(self,image,target=None):
        image = F.adjust_gamma(image,self.gamma)
        if target is not None:
            return image, target
        else:
            return image

if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    img = Image.open('123.tif').convert('RGB')
    ow, oh = img.size
    padh = 576 - oh
    padw = 576 - ow
    img = F.pad(img, (0, 0, padw, padh), fill=0)
    img = np.array(img)[...,0]
    print(img.shape)
    plt.imshow(img)
    plt.show()