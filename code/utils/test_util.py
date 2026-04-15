import cv2
import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def test_all_case(model, test_loader, configs, nms=False):
    total_metric = 0.0
    ith = 0
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(test_loader):
            image = sampled_batch['image']
            label = sampled_batch['label']

            image = image.cpu().data.numpy()
            label = label.cpu().data.numpy()
            label = label.squeeze(0)
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(configs.device), label_batch.to(configs.device)
            if configs.model == 'Vnet':
                outputs_tanh, outputs = model(volume_batch)
            else:
                outputs_tanh, outputs = model(volume_batch)


            outputs_soft = torch.sigmoid(outputs)
            outputs_soft = outputs_soft.cpu().data.numpy()
            outputs_soft = outputs_soft[0, :, :, :]
            score_map = outputs_soft

            prediction = (score_map[0] > 0.5).astype(np.int)

            if nms:
                prediction = getLargestCC(prediction)

            if np.sum(prediction) == 0:
                # single_metric = (0, 0, 0, 0)
                single_metric = (0, 0, 0, 0, 0, 0, 0)
            else:
                single_metric = calculate_metric_percase(prediction, label[:])

            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

            total_metric += np.asarray(single_metric)

            if configs.save_pred_results:
                # 不知道这些数据保存是干什么的
                nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), configs.test_pred_save_path + "/%02d_pred.nii.gz" % ith)
                nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), configs.test_pred_save_path + "/%02d_img.nii.gz" % ith)
                nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), configs.test_pred_save_path + "/%02d_gt.nii.gz" % ith)

                # 保存预测的分割结果
                img_org = sampled_batch['image_org'].cpu().data.numpy().squeeze(0)
                cv2.imwrite('{}/{}.jpg'.format(configs.pred_imgs_save_path, i_batch), img_org)
                prediction = prediction.astype(np.uint8)
                mask = cv2.resize(prediction, (img_org.shape[1], img_org.shape[0]), interpolation=cv2.INTER_LINEAR)

                mask = mask.astype(bool)
                img_org[mask] = img_org[mask] * (1 - configs.alpha) + configs.color_mask * configs.alpha
                cv2.imwrite('{}/{}_mask.jpg'.format(configs.pred_imgs_save_path, i_batch), img_org)

            ith += 1



    avg_metric = total_metric / len(test_loader)
    print('\naverage metric is:\ndice: {:4f}\njc: {:4f}\nhd: {:4f}\nasd: {:4f}'.format(avg_metric[0], avg_metric[1],
                                                                                       avg_metric[2], avg_metric[3]))

    print('\naverage metric is:\ndice: {:4f}\njc: {:4f}\nhd: {:4f}\nasd: {:4f}'
          '\nTPR: {:4f}\nFPR: {:4f}\nAER: {:4f}'.format(avg_metric[0], avg_metric[1], avg_metric[2], avg_metric[3], avg_metric[4], avg_metric[5], avg_metric[6]))

    return avg_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1_tanh, y1 = net(test_patch)
                    # ensemble
                    y = torch.sigmoid(y1)
                    dis_to_mask = torch.sigmoid(-1500*y1_tanh)

                y = y.cpu().data.numpy()
                dis2mask = dis_to_mask.cpu().data.numpy()
                y = y[0, :, :, :, :]
                dis2mask = dis2mask[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = (score_map[0] > 0.5).astype(np.int)


    return label_map, score_map


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
            (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_tpr_fpr(actual, predicted):
    # 计算TPR和FPR
    # 真实类别为1表示正样本，0表示负样本
    # 预测类别也为1或0

    # 计算真正例和假正例
    true_positives = np.sum((actual == 1) & (predicted == 1))
    false_positives = np.sum((actual == 0) & (predicted == 1))


    # 计算TPR和FPR
    if true_positives + false_positives > 0:
        tpr = true_positives / np.sum(actual == 1)  # 真正例率
        fpr = false_positives / (false_positives+np.sum(actual == 0))  # 假正例率
    else:
        tpr = 0.0
        fpr = 0.0

    # 计算AER
    absolute_error = np.sum(predicted != actual)  # 绝对错误，预测和实际值不同的数量
    aer = absolute_error / actual.size

    return tpr, fpr, aer

def calculate_metric_percase(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    # 计算TPR
    true_positives = np.sum((gt == 1) & (pred == 1))
    possible_positives = np.sum(gt == 1)
    true_positive_rate = true_positives / possible_positives if possible_positives > 0 else 0

    # 计算FPR
    false_positives = np.sum((gt == 0) & (pred == 1))
    possible_negatives = np.sum(pred == 0)
    false_positive_rate = false_positives / possible_negatives if possible_negatives > 0 else 0

    tpr, fpr, aer = calculate_tpr_fpr(gt, pred)
    return dice, jc, hd, asd, tpr, fpr, aer
