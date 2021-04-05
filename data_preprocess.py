from torch.utils import data
import torch
import glob
import random
import os
import numpy as np
import SimpleITK as sitk
import skimage.transform as transform
import matplotlib.pyplot as plt


def read_mhd(file_name):
    '''
    根据文件路径读取mhd文件，并返回其内容array、origin、spacing
    '''
    itkimage = sitk.ReadImage(file_name)
    image_array = sitk.GetArrayFromImage(itkimage)
    origin = itkimage.GetOrigin()
    spacing = itkimage.GetSpacing()
    return image_array, origin, spacing


def split(scan):
    result = [0, scan.shape[0] - 1]
    for z in range(scan.shape[0] - 1, -1, -1):
        image = scan[z, :, :]
        image = image / 255
        image = image.astype(np.uint8)
        image = 255 - image
        if np.sum(image) > 3000000:
            result[1] = z
            break
    for index in range(80, 200):
        image = scan[:, :, index]
        image = image / 255
        image = image.astype(np.uint8)
        image = 255 - image
        if np.sum(image) < 4000000:
            continue
        arr = np.sum(image, axis=1)
        arr[arr < 10000] = 0
        zeroLoc = np.where(arr == 0)
        if len(zeroLoc[0]) == 0:
            continue
        if zeroLoc[0][0] + 15 >= len(arr):
            continue
        if arr[zeroLoc[0][0] + 2] == 0 and arr[zeroLoc[0][0] + 15] > 10000:
            if zeroLoc[0][0] < 10:
                continue
            elif arr[zeroLoc[0][0] - 8] > 10000:
                result[0] = (int)(scan.shape[0] - (scan.shape[0] - (zeroLoc[0][0] + 3)) * 0.7)
                return result
            else:
                continue
        else:
            continue
    return result


def generate_heatmap(mask, sigma):
    '''
    :param mask:
    :param sigma:  设置为8,以这两个标记点为中心，生成一个边长为2*sigma+1的正方形，正方形内像素填充为1
    :return:  将mask中标记的点扩充成一个区域
    '''
    for i in range(mask.shape[0]):
        if np.max(mask[i, :, :]) == 0:
            continue
        # 找出当前层不为0的点的坐标，赋值为1,维度顺序Z,Y,X
        index = np.where(mask[i, :, :] != 0)
        for x, y in zip(index[0], index[1]):
            mask[i, max(0, x - sigma):min(511, x + sigma), max(0, y - sigma):min(511, y + sigma)] = 1
    return mask


def normalize(img, mask, window_min, window_max):
    '''
    :param img:
    :param mask:
    :param window_min: 窗口的最小值
    :param window_max: 窗口的最大值
    :return: 归一化成0-1范围的image(float32)和mask(uint8),维度顺序都为Z,Y,X
    '''
    img = img.astype(dtype=np.float32)
    img[img < window_min] = window_min  # 小于窗口最小值置0
    img[img > window_max] = window_max  # 大于窗口最大值置为300
    img = (img - window_min) / (window_max - window_min)
    img = np.array(img, dtype=np.float32)  # 结果保存为float32
    # mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))  # mask直接转换成二值图
    # mask = np.array(mask, dtype=np.uint8)  # 结果保存为uint8
    return img, mask


def image_resize(img, mask):
    '''
    将数据resize成512*512大小,并将img保存成float32类型，mask保存成uint8类型
    '''
    # print(mask.dtype, mask.min(), mask.max(), mask.shape)
    if img.shape[1] != 512 or img.shape[2] != 512:
        img = transform.resize(img, (img.shape[0], 512, 512), mode='constant', anti_aliasing=True, preserve_range=True)
        mask = transform.resize(mask, (mask.shape[0], 512, 512), mode='constant', anti_aliasing=True, preserve_range=True)
    # print(mask.dtype, mask.min(), mask.max())
    img = np.array(img, dtype=np.float32)
    mask = np.array(mask, dtype=np.uint8)
    return img, mask


if __name__ == "__main__":
    # images_ori_dir = '/mnt/data1/wx/data/val/images'
    # masks_ori_dir = '/mnt/data1/wx/data/val/masks'
    # iamges_save_dir = '/mnt/data1/wx/data/val_npy/images'
    # masks_save_dir = '/mnt/data1/wx/data/val_npy/masks'

    # images_ori_dir = '/mnt/data1/wx/data/train/images'
    # masks_ori_dir = '/mnt/data1/wx/data/train/line_masks'
    # iamges_save_dir = '/mnt/data1/wx/data/train_npy/images'
    # masks_save_dir = '/mnt/data1/wx/data/train_npy/line_masks'

    # images_ori_dir = '/mnt/data1/wx/data/val/images'
    # masks_ori_dir = '/mnt/data1/wx/data/val/line_masks'
    # iamges_save_dir = '/mnt/data1/wx/data/val_npy/images'
    # masks_save_dir = '/mnt/data1/wx/data/val_npy/line_masks'

    # images_ori_dir = '/mnt/data1/wx/data/train/images'
    # masks_ori_dir = '/mnt/data1/wx/data/train/seg_line_masks'
    # iamges_save_dir = '/mnt/data1/wx/data/train_npy/images'
    # masks_save_dir = '/mnt/data1/wx/data/train_npy/seg_line_masks'

    images_ori_dir = '/mnt/data1/wx/data/val/images'
    masks_ori_dir = '/mnt/data1/wx/data/val/seg_line_masks'
    iamges_save_dir = '/mnt/data1/wx/data/val_npy/images'
    masks_save_dir = '/mnt/data1/wx/data/val_npy/seg_line_masks'

    if not os.path.exists(masks_save_dir):
        os.makedirs(masks_save_dir)
    
    if not os.path.exists(iamges_save_dir):
        os.makedirs(iamges_save_dir)

    train_img_paths = glob.glob(images_ori_dir + '/*.mhd')
    train_mask_paths = train_img_paths.copy()
    for i in range(len(train_mask_paths)):
        train_mask_paths[i] = train_mask_paths[i].replace(images_ori_dir, masks_ori_dir)
        train_mask_paths[i] = train_mask_paths[i].replace('.mhd', '_seg.mhd')
        # print(train_mask_paths[i])
        # print(train_img_paths[i])
        img, _, _ = read_mhd(train_img_paths[i])  # Z,Y,X,int16
        mask, _, _ = read_mhd(train_mask_paths[i])  # Z,Y,X,uint16
        # 如果大于100层则调用split函数来找目标层
        if img.shape[0] > 100:
            splice_i = split(img)
            # print(str(splice_i[0]) + '   ' + str(splice_i[1]))  # 返回值是一个层数区间
            img = img[splice_i[0]:splice_i[1], :, :]
            mask = mask[splice_i[0]:splice_i[1], :, :]
        img, mask = normalize(img, mask, 0, 300)  # 归一化到0-1的范围,暂时设置窗口值为0-300
        mask = generate_heatmap(mask, 8)  # 先将mask标记的点8邻域扩充为小正方形
        img, mask = image_resize(img, mask)  # 将img,mask,resize成512*512
        iamge_file_id = os.path.relpath(train_img_paths[i], images_ori_dir).split('.')[0]  # 取文件序号,mask与image序号是一一对应的
        # imageSavePath = os.path.join(iamges_save_dir, '{}.npy'.format(iamge_file_id))  # 拼接images的npy后缀的绝对路径
        maskSavePath = os.path.join(masks_save_dir, '{}_seg.npy'.format(iamge_file_id))  # 拼接mask的_seg.npy后缀的绝对路径
        # print(imageSavePath)
        print(maskSavePath)
        # 保存成npy格式
        # np.save(imageSavePath, img)  # float32,Z,Y,X
        np.save(maskSavePath, mask)  # uint8,Z,Y,X,二值图





