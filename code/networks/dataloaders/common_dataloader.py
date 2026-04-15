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
from config import Config as configs   # 对应位置导入配置文件




# 数据增强：随机翻转、旋转
def random_rot_flip(sample):
    image, label = sample['image'], sample['label']
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    if len(label)>0:
        label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if len(label)>0:
        label = np.flip(label, axis=axis).copy()
    sample['image'], sample['label'] = image, label
    return sample

# 数据增强：缩放
def resize(sample, resize_size):
    image, label = sample['image'], sample['label']
    image = cv2.resize(image, resize_size, interpolation=cv2.INTER_LINEAR)
    if len(label)>0:
        label = cv2.resize(label, resize_size, interpolation=cv2.INTER_LINEAR)
    sample['image'], sample['label'] = image, label
    return sample

# 数据增强：归一化
def img_normal(sample):
    image, label = sample['image'], sample['label']
    # 图像数据归一化   一定要进行，否则loss可能为负值
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    if len(label)>0:
        label = (label - np.min(label)) / (np.max(label) - np.min(label))
    image = image.transpose(2, 0, 1).astype(np.float32)
    sample['image'], sample['label'] = image, label
    return sample

# 训练数据增强
def Unet_train_data_process(sample, resize_size=(224, 224), idx='default', model_name='Unet'):
    if model_name != 'Unet_un_augument':  # 不增强方案，则不执行
        sample = random_rot_flip(sample)
    sample = resize(sample, resize_size)

    image, label = sample['image'], sample['label']
    cv2.imwrite("{}/{}.jpg".format(configs.debug_save_path, idx), image)
    cv2.imwrite("{}/{}_mask.jpg".format(configs.debug_save_path, idx), label)

    sample = img_normal(sample)
    return sample

# Unet测试数据处理
def Unet_test_data_process(sample, resize_size=(224, 224), idx='default'):
    sample = resize(sample, resize_size)
    sample = img_normal(sample)
    return sample


# Unet推理数据处理
def Unet_infer_img_process(image, resize_size=(224, 224), idx='1'):
    sample = {'image': image, 'label': np.array([])}
    sample = resize(sample, resize_size)
    sample = img_normal(sample)
    image = sample['image']
    return image


class CommonDataset(Dataset):
    """ LA Dataset """
    # 数据加载器初始化
    def __init__(self, base_dir=None, split='train', transform=None, model_name='Unet'):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.split = split
        self.model_name = model_name

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
        pass


# Unet数据集加载器
class UnetDataset(CommonDataset):
    """ LA Dataset """

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

        if self.split == 'train':
            sample = Unet_train_data_process(sample, resize_size=(224, 224), idx=idx, model_name=self.model_name)
        else:
            sample = Unet_test_data_process(sample, resize_size=(224, 224), idx=idx)
        print(sample['image'].shape)
        return sample



def to_tensor_Vnet(sample):
    image = sample['image']
    label = sample['label']

    # 图像数据归一化  gxz !!!!!!!!!!!!! 一定要进行，否则loss可能为负值
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    if len(label)>0:
        label = (label - np.min(label)) / (np.max(label) - np.min(label))

    image = image.transpose(2, 0, 1).astype(np.float32)
    # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)  # gxz
    # sample['label'] = cv2.cvtColor(sample['label'], cv2.COLOR_BGR2GRAY)
    if 'onehot_label' in sample:
        sample['image'] = torch.from_numpy(image)
        if len(label) > 0:
            sample['label'] = torch.from_numpy(label).long()
        sample['onehot_label'] = torch.from_numpy(sample['onehot_label']).long()
        return sample
    else:
        sample['image'] = torch.from_numpy(image)
        if len(label) > 0:
            sample['label'] = torch.from_numpy(label).long()
        return sample

# Vnet训练数据增强
def Vnet_train_data_process(sample, resize_size=(224, 224), idx='default'):
    sample = random_rot_flip(sample)
    sample = resize(sample, resize_size)

    image, label = sample['image'], sample['label']
    cv2.imwrite("{}/{}.jpg".format(configs.debug_save_path, idx), image)
    cv2.imwrite("{}/{}_mask.jpg".format(configs.debug_save_path, idx), label)

    sample = to_tensor_Vnet(sample)
    return sample

# Vnet测试数据处理
def Vnet_test_data_process(sample, resize_size=(224, 224), idx='default'):
    sample = resize(sample, resize_size)
    sample = to_tensor_Vnet(sample)
    return sample

# Unet推理数据处理
def Vnet_infer_img_process(image, resize_size=(224, 224), idx='1'):
    sample = {'image': image, 'label': np.array([])}
    sample = resize(sample, resize_size)
    sample = to_tensor_Vnet(sample)
    image = sample['image']
    return image

# Vnet数据集加载器
class VnetDataset(CommonDataset):
    """ LA Dataset """

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
        if self.split=='train':
            sample = Vnet_train_data_process(sample, resize_size=(224, 224), idx=idx)
        else:
            sample = Vnet_test_data_process(sample, resize_size=(224, 224), idx=idx)

        return sample

