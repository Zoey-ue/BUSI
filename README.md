# BUSI: Semi-Supervised Breast Ultrasound Image Segmentation

This repository contains a semi-supervised learning project for breast ultrasound image segmentation.  
The project is based on the **Dual-task Consistency Regulation** framework and improves the original method by:

- replacing the original **3D V-Net** backbone with a **2D U-Net**
- integrating a **CBAM (Convolutional Block Attention Module)**
- applying **data augmentation** and **image normalization**

The goal is to improve lesion segmentation accuracy and boundary delineation under limited labeled data settings.

---

## Project Overview

Breast ultrasound image segmentation is an important task in medical image analysis, but it is challenging because:

- lesion boundaries are often ambiguous
- ultrasound images usually contain noise and low contrast
- labeled medical images are expensive and difficult to obtain

To address these issues, this project adopts a **semi-supervised segmentation framework** that leverages both labeled and unlabeled data.  
The baseline is the **Dual-task Consistency Regulation** method, where:

- one branch predicts **pixel-wise probability maps**
- the other branch predicts **level set representations**
- a consistency loss is used to encourage both tasks to produce compatible segmentation outputs

This project adapts the original framework to **2D breast ultrasound segmentation**, improves the network design, and evaluates the method on public breast ultrasound datasets.

---

## Main Contributions

- Reproduced and adapted the **Dual-task Consistency Regulation** framework for breast ultrasound image segmentation
- Replaced the original **3D V-Net** backbone with **2D U-Net** to better fit 2D ultrasound images
- Added **CBAM attention** to enhance spatial and channel-wise feature representation
- Applied **data augmentation** (rotation and flipping) and **image normalization**
- Trained and compared **six different model variants**
- Evaluated performance on **BUSI** and **BUSC / Mendeley ultrasound** datasets
- Compared the proposed method with other breast ultrasound segmentation approaches

---

## Method

### Baseline
The original baseline is a semi-supervised dual-task framework with:

- a **pixel-wise segmentation branch**
- a **level-set function branch**
- a **dual-task consistency loss**

### Improvements in This Project
The proposed model includes the following modifications:

1. **Backbone replacement**
   - original backbone: 3D V-Net
   - modified backbone: 2D U-Net

2. **Attention mechanism**
   - CBAM is added to improve feature extraction
   - both channel attention and spatial attention are used

3. **Data preprocessing**
   - image normalization to `[0, 1]`
   - data augmentation including rotation and flipping

---

## Datasets

### 1. BUSI Dataset
The Breast Ultrasound Images (BUSI) dataset is used for training and testing.

- Total images: **780**
- Categories: **benign, malignant, normal**
- Testing images: **186**
- Default batch size: **8**
- Labeled / unlabeled ratio: **2 labeled + 6 unlabeled**

### 2. BUSC / Mendeley Ultrasound Dataset
An additional breast ultrasound dataset is used for testing generalization performance.

- **100 benign** images
- **150 malignant** images
- segmentation annotations provided by radiologists in prior work

---

## Experimental Setup

- Framework: **PyTorch**
- Python: **3.6**
- Main libraries: **NumPy, scikit-image, SimpleITK, SciPy**
- Training iterations: **10,000**
- Initial learning rate: **0.005**
- Learning rate decay: multiply by **0.1 every 2500 iterations**
- Hardware:
  - Intel Core i9-12900H
  - NVIDIA GeForce RTX 3070 Ti

---

## Evaluation Metrics

The project uses the following segmentation metrics:

- **DSC (Dice Similarity Coefficient)**
- **JSC (Jaccard Similarity Coefficient)**
- **95HD (95% Hausdorff Distance)**
- **TPR (True Positive Rate)**

---

## Model Variants

Six models were trained and compared:

- **Model 1**: Replace 3D convolutions with 2D convolutions only
- **Model 2**: Model 1 + data augmentation
- **Model 3**: Replace backbone with U-Net
- **Model 4**: Model 3 + data augmentation + normalization
- **Model 5**: Model 4 + CBAM (**proposed method**)
- **Model 6**: Same as Model 5, but trained with 50% labeled data instead of 25%

---

## Results

### BUSI Dataset

| Model | DSC ↑ | JSC ↑ | 95HD ↓ |
|------|------:|------:|-------:|
| Model 1 | 0.8046 | 0.7043 | 17.5577 |
| Model 2 | 0.8168 | 0.7182 | 15.6363 |
| Model 3 | 0.8338 | 0.7411 | 16.5844 |
| Model 4 | 0.8372 | 0.7426 | 16.7550 |
| Model 5 | **0.8604** | **0.7681** | 15.2752 |
| Model 6 | 0.8502 | 0.7643 | **14.1095** |

**Key findings on BUSI:**
- The proposed method (**Model 5**) achieved the best overall segmentation performance
- Compared with baseline, it improved:
  - **DSC by 5.58%**
  - **JSC by 6.39%**
  - **95HD by 2.28**

### BUSC Dataset

| Model | DSC ↑ | JSC ↑ | 95HD ↓ |
|------|------:|------:|-------:|
| Model 1 | 0.7501 | 0.6169 | 26.0910 |
| Model 2 | 0.7045 | 0.5784 | 26.2522 |
| Model 3 | **0.7541** | **0.6307** | **23.2630** |
| Model 4 | 0.7426 | 0.6141 | 24.8066 |
| Model 5 | 0.7521 | 0.6242 | 26.3530 |
| Model 6 | 0.7208 | 0.5890 | 25.8747 |

**Observation:**
- The proposed improvements were most effective on the **BUSI dataset**
- On the BUSC dataset, **Model 3** performed best, possibly due to differences between training and testing distributions

---

## Comparison with Other Methods

On the **BUSI dataset**, the proposed model outperformed several existing methods and achieved:

- **TPR: 0.8603**
- **DSC: 0.8604**
- **JSC: 0.7681**

Compared with traditional **U-Net**, the proposed method showed substantial gains.

On the **BUSC dataset**, the proposed method also outperformed traditional U-Net and several other baselines in DSC and JSC, while PDF-UNet remained competitive in some settings.

---
## References

Original baseline: [Semi-supervised Medical Image Segmentation through Dual-task Consistency](https://cdn.aaai.org/ojs/17066/17066-13-20560-1-2-20210518.pdf)
## Code Acknowledgement

Part of this implementation is adapted from the original publicly available codebase of the Dual-task Consistency Regulation framework.  
All credit for the baseline framework belongs to the original authors.

# BUSI：基于半监督学习的乳腺超声图像分割

本仓库为一个面向**乳腺超声图像分割**的半监督学习项目。

本项目基于 **Dual-task Consistency Regulation（双任务一致性约束）** 框架，并在原始方法基础上进行了以下改进：

- 将原始 **3D V-Net** 主干网络替换为 **2D U-Net**
- 引入 **CBAM（卷积块注意力模块）**
- 加入**数据增强**与**图像归一化**预处理策略

项目目标是在**标注数据有限**的情况下，提高乳腺病灶分割的精度与边界识别能力。

---

## 项目简介

乳腺超声图像分割是医学图像分析中的重要任务，但同时也具有较大挑战，主要原因包括：

- 病灶边界通常较为模糊
- 超声图像普遍存在噪声大、对比度低的问题
- 医学图像标注成本高，获取难度大

为了解决上述问题，本项目采用了一个同时利用**有标注数据**与**无标注数据**的半监督分割框架。

本项目所采用的 baseline 为 **Dual-task Consistency Regulation** 方法，其核心思想为：

- 一个分支用于预测**像素级概率图**
- 另一个分支用于预测**水平集表示（level-set representation）**
- 通过一致性损失约束两个任务输出的分割结果保持一致

在此基础上，本项目将原始框架适配到**二维乳腺超声图像分割任务**中，并对网络结构与预处理流程进行了改进，同时在公开乳腺超声数据集上进行了实验验证。

---

## 项目贡献

- 复现并适配 **Dual-task Consistency Regulation** 半监督分割框架至乳腺超声图像分割任务
- 使用 **2D U-Net** 替代原始 **3D V-Net**，以更好适配二维超声图像
- 引入 **CBAM 注意力机制**，增强空间与通道特征表达能力
- 加入**数据增强**（旋转、翻转）与**图像归一化**操作
- 训练并比较了 **6 种不同模型变体**
- 在 **BUSI** 与 **BUSC / Mendeley ultrasound** 数据集上进行了实验评估
- 将改进方法与其他乳腺超声图像分割方法进行了对比分析

---

## 方法说明

### Baseline
原始 baseline 是一个半监督双任务分割框架，主要包含：

- **像素级分割分支**
- **水平集函数分支**
- **双任务一致性损失**

### 本项目的改进点

1. **主干网络替换**
   - 原始主干：**3D V-Net**
   - 改进后主干：**2D U-Net**

2. **注意力机制引入**
   - 在网络中加入 **CBAM**
   - 同时利用**通道注意力**与**空间注意力**提升特征提取能力

3. **数据预处理**
   - 将图像像素归一化到 `[0, 1]`
   - 使用旋转与翻转进行数据增强

---

## 数据集

### 1. BUSI 数据集
BUSI（Breast Ultrasound Images）数据集用于训练与测试。

- 图像总数：**780**
- 类别：**良性（benign）、恶性（malignant）、正常（normal）**
- 测试集图像数：**186**
- 默认 batch size：**8**
- 有标注 / 无标注比例：**2 张有标注图像 + 6 张无标注图像**

### 2. BUSC / Mendeley Ultrasound 数据集
另一个乳腺超声数据集用于测试模型泛化能力。

- **100 张良性图像**
- **150 张恶性图像**
- 分割标注由相关工作中的放射科医生提供

---

## 实验设置

- 深度学习框架：**PyTorch**
- Python 版本：**3.6**
- 主要依赖库：**NumPy、scikit-image、SimpleITK、SciPy**
- 训练迭代次数：**10,000**
- 初始学习率：**0.005**
- 学习率衰减策略：每 **2500** 次迭代乘以 **0.1**
- 实验硬件：
  - Intel Core i9-12900H
  - NVIDIA GeForce RTX 3070 Ti

---

## 评价指标

本项目使用以下指标评估分割效果：

- **DSC（Dice Similarity Coefficient，Dice 相似系数）**
- **JSC（Jaccard Similarity Coefficient，Jaccard 相似系数）**
- **95HD（95% Hausdorff Distance）**
- **TPR（True Positive Rate，真正率）**

---

## 模型变体

本项目共训练并比较了 6 种模型：

- **Model 1**：仅将 3D 卷积替换为 2D 卷积
- **Model 2**：在 Model 1 基础上加入数据增强
- **Model 3**：将 backbone 替换为 U-Net
- **Model 4**：在 Model 3 基础上加入数据增强与归一化
- **Model 5**：在 Model 4 基础上加入 CBAM（**本文提出的方法**）
- **Model 6**：与 Model 5 结构相同，但使用 50% 有标注数据训练，而非 25%

---

## 实验结果

### BUSI 数据集

| 模型 | DSC ↑ | JSC ↑ | 95HD ↓ |
|------|------:|------:|-------:|
| Model 1 | 0.8046 | 0.7043 | 17.5577 |
| Model 2 | 0.8168 | 0.7182 | 15.6363 |
| Model 3 | 0.8338 | 0.7411 | 16.5844 |
| Model 4 | 0.8372 | 0.7426 | 16.7550 |
| Model 5 | **0.8604** | **0.7681** | 15.2752 |
| Model 6 | 0.8502 | 0.7643 | **14.1095** |

**BUSI 数据集结论：**
- 本项目提出的方法（**Model 5**）取得了最佳整体分割效果
- 相较 baseline，性能提升如下：
  - **DSC 提升 5.58%**
  - **JSC 提升 6.39%**
  - **95HD 改善 2.28**

### BUSC 数据集

| 模型 | DSC ↑ | JSC ↑ | 95HD ↓ |
|------|------:|------:|-------:|
| Model 1 | 0.7501 | 0.6169 | 26.0910 |
| Model 2 | 0.7045 | 0.5784 | 26.2522 |
| Model 3 | **0.7541** | **0.6307** | **23.2630** |
| Model 4 | 0.7426 | 0.6141 | 24.8066 |
| Model 5 | 0.7521 | 0.6242 | 26.3530 |
| Model 6 | 0.7208 | 0.5890 | 25.8747 |

**结果观察：**
- 本项目提出的改进方法在 **BUSI 数据集** 上效果最明显
- 在 **BUSC 数据集** 上，**Model 3** 表现最好，可能与训练集和测试集之间的数据分布差异有关

---

## 与其他方法的比较

在 **BUSI 数据集** 上，本项目方法取得了较强的分割性能：

- **TPR：0.8603**
- **DSC：0.8604**
- **JSC：0.7681**

与传统 **U-Net** 相比，本项目方法取得了明显提升。

在 **BUSC 数据集** 上，本项目方法同样在 DSC 和 JSC 指标上优于传统 U-Net 及部分其他基线方法，但在部分设置下，PDF-UNet 仍具有一定竞争力。


