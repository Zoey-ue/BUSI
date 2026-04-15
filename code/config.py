# -*- coding: utf-8 -*-


import torch
import numpy as np
from utils.file_tools import check_exit_dir

class Config:
	model = "Unet_CBAM"         # 可以选择以下模型 Vnet、Vnet_Aug、Unet、Unet_Aug、Unet_Aug84、Unet_CBAM

	data_root_path = r'./data/BUSI/'  # BUSI训练、测试数据集存放路径
	# data_root_path = r'./data/BUSC/'  # BUSC测试数据集存放路径


	save_pred_results = True   # 测试时是否保存预测图片
	pred_imgs_save_path = r'./output/output_imgs'  # 测试图片保存路径
	test_pred_save_path = r'./output/output_infos'  # 测试信息保存路径
	debug_save_path = r'./output/debug'  # 测试信息保存路径

	# 是否使用gpu，自动识别，无gpu情况自动切换到cpu
	device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

	classes_num = 1              # 输出类别数，本数据集只有1类
	resize_size = (224, 224)     # 训练和识别时将图片缩放到该尺寸


	# 各模型存放路径  默认batch_size为8，有标签数据为2
	model_Vnet_path       = r'./train/Vnet_0.3_8_2/iter_10000.pth'  # Vnet模型存放路径
	model_Vnet_Aug_path   = r'./train/Vnet_Aug_0.3_8_2/iter_10000.pth'  # Vnet数据增强模型存放路径
	model_Unet_path       = r'./train/Unet_0.3_8_2/iter_10000.pth'  # Unet模型存放路径
	model_Unet_Aug_path   = r'./train/Unet_Aug_0.3_8_2/iter_10000.pth'  # Unet数据增强操作模型存放路径
	model_Unet_Aug84_path = r'./train/Unet_Aug_0.3_8_4/iter_10000.pth'  # Unet batch_size8 有监督数据4
	model_Unet_CBAM_path  = r'./train/Unet_CBAM_0.3_8_2/iter_10000.pth'  # Unet_CBAM模型存放路径

	if model == "Vnet":
		model_path = model_Vnet_path
	elif model == "Vnet_Aug":
		model_path = model_Vnet_Aug_path
	elif model == "Unet":
		model_path = model_Unet_path
	elif model == "Unet_Aug":
		model_path = model_Unet_Aug_path
	elif model == "Unet_Aug84":
		model_path = model_Unet_Aug84_path
	elif model == "Unet_CBAM":
		model_path = model_Unet_CBAM_path
	else:
		model = "Unet"
		model_path = model_Unet_Aug_path

	# 分割填充颜色、透明度
	color_mask = np.array(list((125, 0, 180)))   # 分割填充颜色
	# color_mask = np.array(list((64, 255, 0)))   # 分割填充颜色
	alpha = 0.5                                  # 分割填充透明度


	# 其他一些不重要的配置
	n_channels = 3               # 输入图片的通道数
	bilinear = False             # Unet网络上采样时使用使用双线性差值 'Use bilinear upsampling'
	normalization = 'batchnorm'  #
	has_dropout = False


	if save_pred_results:
		check_exit_dir(pred_imgs_save_path)
		check_exit_dir(test_pred_save_path)
		check_exit_dir(debug_save_path)




