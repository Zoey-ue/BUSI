# -*- coding: utf-8 -*-

import os
import shutil

img_mask_dir = r'D:\Project2\DTC-master\data\BUSC\test\masks'
img_gt_dir = r'D:\Project2\DTC-master\data\BUSC\test\gt'

for file in os.listdir(img_mask_dir):
    img_old_path = os.path.join(img_mask_dir, file)
    new_file_name = os.path.splitext(file)[0]+'_anno'+os.path.splitext(file)[1]
    img_new_path = os.path.join(img_gt_dir, new_file_name)
    shutil.copy(img_old_path, img_new_path)
