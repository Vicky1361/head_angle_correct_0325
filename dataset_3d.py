import torch
from torch.utils import data
import numpy as np
import random
from scipy import ndimage
from rotate_3d import rotate_img_seg


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


def addNoise(img):
    """给数据添加噪声

	:param img: 输入的原数据
	:return: 添加了噪声的数据
	"""
    for i in range(0, 5):
        random_index = np.random.randint(0, img.shape[0] - 1)
        img[random_index, :, :] = img[random_index, :, :] + np.random.normal(0.0, 0.2,
                                                                             size=(img.shape[1], img.shape[2]))
    return img


from time import time


class my_dataset_seg(data.Dataset):
    def __init__(self, img_paths, mask_paths, slice_nums, is_crop=True, is_rotate=True, is_gray_scale_shift=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.slice_nums = slice_nums
        self.is_crop = is_crop
        self.is_rotate = is_rotate
        self.is_gray_scale_shift = is_gray_scale_shift

    def __getitem__(self, index):
        # startTime = time()
        img = np.load(self.img_paths[index])
        mask = np.load(self.mask_paths[index])
        # endTime = time()
        # print('load use', endTime - startTime, 's')
        # print('img', img.min(), img.max(), img.dtype)
        # print('mask', mask.min(), mask.max(), mask.dtype)
        # 数据在线增强
        # startTime = time()
        img, mask = self.image_enhancement(img, mask)
        # endTime = time()
        # print('image_enhancement use', endTime - startTime, 's')

        # 转为float tensor类型再返回
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return img, mask

    def __len__(self):
        return len(self.img_paths)

    def image_enhancement(self, img, mask):
        """数据在线增强

		:param img: 输入的原数据
		:param mask: 输入的原数据对应的mask
		:return: 增强后得到的数据和mask
		"""

        # 层间采样并添加噪声
        # startTime = time()
        splice_i = [0, img.shape[0] - 1]
        splice_index = getMoreDataIndex(splice_i, np.random.randint(0, 6))
        img = img[splice_index]
        mask = mask[splice_index]
        # endTime = time()
        # print('getMoreDataIndex use', endTime - startTime, 's')

        # startTime = time()
        # 统一数据层数为slice_nums
        if img.shape[0] > self.slice_nums:
            begin = random.randint(0, img.shape[0] - self.slice_nums)
            img = img[begin:begin + self.slice_nums]
            mask = mask[begin:begin + self.slice_nums]
            img = addNoise(img)
        elif img.shape[0] < self.slice_nums:
            img = addNoise(img)
            img = np.pad(img, [(0, self.slice_nums - img.shape[0]), (0, 0), (0, 0)], mode='constant', constant_values=0)
            mask = np.pad(mask, [(0, self.slice_nums - mask.shape[0]), (0, 0), (0, 0)], mode='constant',constant_values=0)
        else:
            pass

        if self.is_gray_scale_shift:
            # print('1', img.dtype, img.max(), img.min())
            gray_scale = np.random.random() * 0.2 + 0.9
            gray_shift = np.random.random() * 0.2 - 0.1
            img -= 0.5  # (0~1) -> (-0.5~0.5)
            img *= gray_scale
            img += gray_shift
            img = np.clip(img, -0.5, 0.5)
            img += 0.5
        # print('2', img.dtype, img.max(), img.min())

        # endTime = time()
        # print('change slices num use', endTime - startTime, 's')
        # 裁剪暂时不做
        # if self.is_crop == True:
        #     left = random.randint(0, 10)
        #     right = random.randint(0, 10)
        #     up = random.randint(0, 10)
        #     down = random.randint(0, 10)
        #     img = img[:, left:img.shape[1]-right, down:img.shape[2]-up]
        #     mask = mask[:, left:mask.shape[1]-right, down:mask.shape[2]-up]

        # 随机旋转
        if self.is_rotate == True:
            # startTime = time()
            rotate_max_degree = 30  # 旋转角度范围是[-rotate_max_degree,rotate_max_degree]
            random_rotate = (random.random() - 0.5) * 2 * rotate_max_degree
            img = ndimage.rotate(img, random_rotate, axes=(2, 1), order=0, reshape=False)
            mask = ndimage.rotate(mask, random_rotate, axes=(2, 1), order=0, reshape=False)
        # img = np.transpose(img, [1, 2, 0])
        # mask = np.transpose(mask, [1, 2, 0])
        # img, mask = rotate_img_seg(img, mask, random.random() - 0.5)
        # img = np.transpose(img, [2, 0, 1])
        # mask = np.transpose(mask, [2, 0, 1])
        # endTime = time()
        # print('rotate use', endTime - startTime, 's')
        return img, mask
