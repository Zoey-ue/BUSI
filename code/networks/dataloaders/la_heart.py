import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler

class LAHeart(Dataset):
    """ LA Dataset """
    # 数据加载器初始化
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        img_dir = os.path.join('{}/{}'.format(self._base_dir, split), 'img')   # 图片对应的文件夹
        gt_dir = os.path.join('{}/{}'.format(self._base_dir, split), 'gt')     # 标注数据对应的文件夹

        img_file_list = os.listdir(img_dir)
        gt_file_list  = os.listdir(gt_dir)

        # 获取所有的图片路径和标签路径，存放到数据索引列表中
        self.img_path_list = [os.path.join(img_dir, file) for file in img_file_list]
        self.gt_path_list = [os.path.join(gt_dir, '{}_anno{}'.format(os.path.splitext(file)[0], os.path.splitext(file)[-1])) for file in img_file_list]

        print("total {} samples".format(len(self.img_path_list)))

    def __len__(self):
        return len(self.img_path_list)

    # 数据加载器逐个加载的方法
    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]   # 图片路径
        gt_path = self.gt_path_list[idx]     # 标签路径

        image = cv2.imread(img_path)                       # 读取图片信息
        # label = cv2.imread(gt_path)  # 读取标签信息
        label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # 读取标签信息
        image_org, label_org = image.copy(), label.copy()
        sample = {'image': image, 'label': label,
                  'image_org': image_org, 'label_org': label_org}

        if self.transform:   # 进行数据增强
            sample = self.transform(sample)

        return sample


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}



class InferResizePast(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size
        self.h_crop = self.output_size[0]
        self.w_crop = self.output_size[1]

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h_img, w_img = image.shape[0], image.shape[1]
        if h_img>self.h_crop or w_img>self.w_crop:
            image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, self.output_size, interpolation=cv2.INTER_LINEAR)
            return {'image': image, 'label': label}
        else:
            img_new = np.zeros((self.h_crop, self.w_crop, 3), dtype=np.uint8)
            label_new = np.zeros((self.h_crop, self.w_crop), dtype=np.uint8)
            img_new[:h_img, :w_img,:] = image[:h_img, :w_img,:]
            label_new[:h_img, :w_img] = label[:h_img, :w_img]
            return {'image': img_new, 'label': label_new}

class RandomCropResize(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size
        self.h_crop = self.output_size[0]
        self.w_crop = self.output_size[1]

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random()>0.3:  # 缩放到指定尺寸
            image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, self.output_size, interpolation=cv2.INTER_LINEAR)

            return {'image': image, 'label': label}
        else:   # 裁减到指定尺寸
            h_img, w_img = image.shape[0], image.shape[1]
            x_range = w_img-self.w_crop if w_img>self.w_crop else 0
            y_range = h_img-self.h_crop if h_img>self.h_crop else 0
            x1, y1, x2, y2 = 0, 0, w_img, h_img

            if x_range>0:  # 计算随机裁减范围
                x1 = random.randint(0, x_range)
                x2 = x1 + self.w_crop
            if y_range>0:
                y1 = random.randint(0, y_range)
                y2 = y1 + self.h_crop

            # 计算贴图范围
            w_tmp, h_tmp = x2-x1, y2-y1
            past_x = random.randint(0, self.w_crop - w_tmp) if self.w_crop > w_tmp else 0
            past_y = random.randint(0, self.h_crop - h_tmp) if self.h_crop > h_tmp else 0

            img_new = np.zeros((self.h_crop, self.w_crop, 3), dtype=np.uint8)
            label_new = np.zeros((self.h_crop, self.w_crop), dtype=np.uint8)

            img_new[past_y:past_y+h_tmp, past_x:past_x+w_tmp,:] = image[y1:y2,x1:x2,:]
            label_new[past_y:past_y+h_tmp, past_x:past_x+w_tmp] = label[y1:y2,x1:x2]
            # cv2.imwrite("image.jpg", img_new)
            # cv2.imwrite("label.jpg", label_new)
            return {'image': img_new, 'label': label_new}

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}

# 新增resize方法
class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size


    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.output_size, interpolation=cv2.INTER_LINEAR)
        sample['image'], sample['label'] = image, label
        return sample

class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, model_name=''):
        self.model_name = model_name

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if 'Aug' in self.model_name:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()

        sample = {'image': image, 'label': label}
        return sample


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        cv2.imwrite("output/{}.jpg".format(1), image)
        cv2.imwrite("output/{}_mask.jpg".format(1), label)
        # 图像数据归一化  gxz !!!!!!!!!!!!! 一定要进行，否则loss可能为负值
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        label = (label - np.min(label)) / (np.max(label) - np.min(label))


        image = image.transpose(2, 0, 1).astype(np.float32)
        # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)  # gxz
        # sample['label'] = cv2.cvtColor(sample['label'], cv2.COLOR_BGR2GRAY)
        if 'onehot_label' in sample:
            sample['image'] = torch.from_numpy(image)
            sample['label'] = torch.from_numpy(label).long()
            sample['onehot_label'] = torch.from_numpy(sample['onehot_label']).long()
            return sample
        else:
            sample['image'] = torch.from_numpy(image)
            sample['label'] = torch.from_numpy(label).long()
            return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices      # 有标签的索引
        self.secondary_indices = secondary_indices  # 无标签的索引
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        # 随机打乱索引顺序
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch for (primary_batch, secondary_batch) in zip(grouper(primary_iter, self.primary_batch_size), grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable) # 返回一个随机排列（重置）给定的副本的副本


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)