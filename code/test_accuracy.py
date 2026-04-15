# -*- coding: utf-8 -*

import torch
from networks.unet.unet_model import UNet
from networks.vnet.vnet import VNet
from networks.dataloaders.common_dataloader import UnetDataset, VnetDataset
from torch.utils.data import DataLoader
from utils.test_util import test_all_case
from torchsummary import summary

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

    # 构建Vnet测试数据集
    db_set = VnetDataset(base_dir=configs.data_root_path, split='test')

    return model, db_set

# 获取Unet模型和数据集
def get_Unet_model():
    if configs.model == 'Unet_CBAM':
        print('加载Unet_CBAM模型，模型路径为{}'.format(configs.model_path))
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

    # 构建Unet测试数据集
    db_set = UnetDataset(base_dir=configs.data_root_path, split='test')

    return model, db_set


def test_calculate_metric():
    # 构建模型
    print("在{}路径加载模型参数".format(configs.model_path))
    if configs.model.startswith('Vnet'):
        model, db_set = get_Vnet_model()
    else:
        model, db_set = get_Unet_model()

    # 打印模型的结构和参数量信息
    summary(model, input_size=(3, 224, 224))
    # 打印网络结构和参数信息
    print(model)

    # 创建数据集加载器
    test_loader = DataLoader(db_set, num_workers=1, pin_memory=True)

    # 对测试集所有的样本进行展示
    avg_metric = test_all_case(model, test_loader, configs)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()  # 6000

