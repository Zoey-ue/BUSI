# -*- coding: utf-8 -*-

import os
import cv2
import torch
import numpy as np
from networks.unet.unet_model import UNet
from networks.vnet.vnet import VNet
from networks.dataloaders.common_dataloader import UnetDataset, VnetDataset
from torch.utils.data import DataLoader
from utils.test_util import test_all_case
from networks.dataloaders.common_dataloader import Vnet_infer_img_process
from networks.dataloaders.common_dataloader import Unet_infer_img_process
from utils.file_tools import check_exit_dir

from config import Config as configs   # 对应位置导入配置文件

# 获取Vnet模型和数据集
def get_Vnet_model():
    print('加载Vnet模型，模型路径为{}'.format(configs.model_path))
    net = VNet(n_channels=configs.n_channels, n_classes=configs.classes_num, normalization=configs.normalization,
               has_dropout=configs.has_dropout)
    # 将模型加载到设备
    model = net.to(configs.device)
    # 加载模型参数
    model.load_state_dict(torch.load(configs.model_path))
    model.eval()

    return model

# 获取Unet模型和数据集
def get_Unet_model():
    print('加载Unet模型，模型路径为{}'.format(configs.model_path))
    if configs.model == 'Unet_CBAM':
        net = UNet(n_channels=configs.n_channels, n_classes=configs.classes_num, bilinear=configs.bilinear, use_CBAM=True)
    else:
        net = UNet(n_channels=configs.n_channels, n_classes=configs.classes_num, bilinear=configs.bilinear)

    # 将模型加载到设备
    model = net.to(configs.device)

    state_dict = torch.load(configs.model_path, map_location=configs.device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print('Ignoring "' + str(e) + '"')
    model.eval()  # 推理模式

    return model

def get_model(model_name):
    print('准备加载模型')
    if model_name.startswith('Vnet'):
        model = get_Vnet_model()
    else:
        model = get_Unet_model()
    return model

# Vnet推理图片处理
def img_to_tensor_Vnet(image):
    # image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    # # 图像数据归一化  gxz !!!!!!!!!!!!! 一定要进行，否则loss可能为负值
    # image = (image - np.min(image)) / (np.max(image) - np.min(image))
    # image = image.transpose(2, 0, 1).astype(np.float32)
    # image = np.expand_dims(image, axis=0)
    # image_tensor = torch.from_numpy(image)

    image = Vnet_infer_img_process(image, resize_size=configs.resize_size)
    image = image.unsqueeze(0)
    image_tensor = image.to(device=configs.device, dtype=torch.float32)

    return image_tensor

# Unet推理图片处理
def img_to_tensor_Unet(image):
    image = torch.from_numpy(Unet_infer_img_process(image, resize_size=configs.resize_size))
    image = image.unsqueeze(0)
    image_tensor = image.to(device=configs.device, dtype=torch.float32)
    return image_tensor

# 模型推理
def model_infer(model, image_tensor, model_name='Vnet'):
    with torch.no_grad():
        image_tensor = image_tensor.to(configs.device)
        if model_name=='Vnet':
            outputs_tanh, outputs = model(image_tensor)
        else:
            outputs_tanh, outputs = model(image_tensor)
        outputs_soft = torch.sigmoid(outputs)
        outputs_soft = outputs_soft.cpu().data.numpy()
        outputs_soft = outputs_soft[0, :, :, :]
        score_map = outputs_soft

        prediction = (score_map[0] > 0.5).astype(np.int)
    return prediction

# 模型推理及结果展示
def infer_and_draw(img_path, model, model_name='Vnet'):
    image = cv2.imread(img_path)
    if model_name=='Vnet':
        image_tensor = img_to_tensor_Vnet(image)
    else:
        image_tensor = img_to_tensor_Unet(image)
    mask_pred = model_infer(model, image_tensor, model_name).astype(np.uint8)
    mask_pred = cv2.resize(mask_pred, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    mask = mask_pred.astype(bool)
    image[mask] = image[mask] * (1 - configs.alpha) + configs.color_mask * configs.alpha
    return image

def show(path, model, model_name, save_dir=r'./output/show'):
    check_exit_dir(save_dir)
    # 路径是一个文件夹
    if os.path.isdir(path):
        print('路径为文件夹，进行全部预测')
        for file in os.listdir(path):
            print('推理图片文件名为{}'.format(file))
            img_path = os.path.join(path, file)
            image = infer_and_draw(img_path, model, model_name)
            cv2.imwrite('{}/{}'.format(save_dir, file), image)
            print('推理完成，结果存放在{}/{}'.format(save_dir, file))
    else:
        print('路径为单个文件')
        image = infer_and_draw(path, model, model_name)
        cv2.imwrite('{}/{}'.format(save_dir, '0_result.jpg'), image)
        print('推理完成，结果存放在{}/{}'.format(save_dir, '0_result.jpg'))
        cv2.imshow('0_result.jpg', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    model = get_model(configs.model)  # 加载模型

    # 路径可以是具体的文件或者文件夹，可自动根据路径进行判断及推理，结果放到save_dir中
    # path = r'.\data\BUSI\test\img\00007.bmp'
    # path = r'.\data\BUSI\test\img'
    path = r'.\data\BUSC\images'

    show(path, model, model_name=configs.model, save_dir=r'./output/show')

