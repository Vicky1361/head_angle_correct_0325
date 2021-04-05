import torch
from torch.utils import data
import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import random
from scipy import ndimage
from rotate_3d import rotate_img_seg


def read_mhd(file_name):
    '''
    根据文件路径读取mhd文件，并返回其内容array、origin、spacing
    '''
    itkimage = sitk.ReadImage(file_name)
    image_array = sitk.GetArrayFromImage(itkimage)
    origin = itkimage.GetOrigin()
    spacing = itkimage.GetSpacing()
    return image_array, origin, spacing


def getMoreDataIndex(splice_i, mode):
    """对数据进行层间采样

	:param splice_i:采样的下标范围
	:param mode:采样模式
	:return:采样得到的下标列表,采样之后得到一定是24层？？？
	"""
    if (splice_i[1] - splice_i[0]) < 48:
        return list(range(splice_i[0], splice_i[1] + 1))
    disc = (splice_i[1] - splice_i[0]) // 24
    index = []
    index_mode = []
    if mode == 0:  # 鼻子及其下部采样多
        index_mode.extend([1, 1, disc // 2, disc // 2 + 1, disc])
    elif mode == 1:  # 中间采样多
        index_mode.extend([disc, disc // 2, 1, disc // 2, disc])
    elif mode == 2:  # 脑部采样多
        index_mode.extend([disc, disc // 2 + 1, disc // 2, 1, 1])
    elif mode == 3:  # 中间采用少，鼻子下部和脑部采样多
        index_mode.extend([1, disc // 2, disc, disc // 2, 1])
    elif mode == 4:  # 等距采样
        index_mode.extend([disc, disc, disc, disc, disc])
    else:  # 随机采样
        for i in range(0, 5):
            index_mode.append(np.random.randint(1, disc))

    index_mode_arr = np.array(index_mode)
    slice_num = (splice_i[1] - splice_i[0]) // np.sum(index_mode_arr)
    for i in range(1, slice_num + 1):
        index.append(splice_i[0] + index_mode_arr[0] * i)
    for i in range(1, slice_num + 1):
        index.append(splice_i[0] + index_mode_arr[0] * slice_num + index_mode_arr[1] * i)
    for i in range(1, slice_num + 1):
        index.append(splice_i[0] + np.sum(index_mode_arr[0:2]) * slice_num + index_mode_arr[2] * i)
    for i in range(1, slice_num + 1):
        index.append(splice_i[0] + np.sum(index_mode_arr[0:3]) * slice_num + index_mode_arr[3] * i)
    for i in range(1, slice_num + 1):
        index.append(splice_i[0] + np.sum(index_mode_arr[0:4]) * slice_num + index_mode_arr[4] * i)
    return index

if __name__ == "__main__":
    train_img_dir = '/mnt/data1/wx/data/train_npy/03_26_images'  # 部分用于测试getMoreDataIndex函数的数据
    img_save_dir = '/mnt/data1/wx/data/train_npy/test_sampling/'
    train_img_paths = glob.glob(train_img_dir + '/*.npy')
    print('len: ', len(train_img_paths))
    for i in range(len(train_img_paths)):
        # img, _, _ = read_mhd(train_img_paths[i])
        # 直接读取npy格式的文件

        if i == 0:
            iamge_file_id = os.path.relpath(train_img_paths[i], train_img_dir).split('.')[0]  # 取文件序号,mask与image序号是一一对应的
            print('当前病例号: ', iamge_file_id)
            img = np.load(train_img_paths[i])
            splice_i = [0, img.shape[0] - 1]
            print('采样前的层数: ', img.shape[0])
            splice_index = getMoreDataIndex(splice_i, np.random.randint(0, 6))
            print('采样后的层数:  ', len(splice_index))  # 这是做的一个随机采样,每一次采样后得到的层数不一定相等
            print('具体有哪些层: ', splice_index)
            img = img[splice_index]


            # 直接可视化npy的每一层
            for k in range(img.shape[0]):
                plt.figure()
                plt.title(str(iamge_file_id) + '_' + str(k))
                plt.imshow(img[k],cmap='gray')
                pic_name = img_save_dir + str(iamge_file_id) + '+' + str(k)
                plt.savefig('%s.jpg' % pic_name)
                plt.close()







        # 想尝试将npy格式采样之后保存为mhd文件
        # 将采样后的图像保存成mhd
        # iamge_file_id = os.path.relpath(train_img_paths[i], train_img_dir).split('.')[0]  # 取文件序号,mask与image序号是一一对应的
        # imgSavePath = os.path.join(img_save_dir, '{}.npy'.format(iamge_file_id))  # 拼接mask的_seg.npy后缀的绝对路径
        # # print(imageSavePath)
        # print('采样后的图像保存的绝对路径:  ', imgSavePath)
        # sitk.WriteImage(img, imgSavePath)







