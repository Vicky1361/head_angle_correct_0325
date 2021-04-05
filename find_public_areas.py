import os
import time
import cv2
from glob import glob
import numpy as np
import SimpleITK as sitk
from skimage.measure import label


def readMhd(mhdPath):
    '''
	根据文件路径读取mhd文件，并返回其内容Array、origin、spacing
	'''
    itkImage = sitk.ReadImage(mhdPath)
    imageArray = sitk.GetArrayFromImage(itkImage)  # [Z, H, W]
    origin = itkImage.GetOrigin()
    spacing = itkImage.GetSpacing()
    return imageArray, origin, spacing


def findCenterPoints(image):
    '''
    :param image: 二维矩阵
    :return: 当前层的中点x,y坐标
    '''
    H = image.shape[0] // 2
    W = image.shape[1] // 2
    centerHW = [int(H), int(W)]  # 取整
    return centerHW


def otsuEnhance(img_gray, th_begin, th_end, th_step=1):
    '''
    根据类间方差最大求最适合的阈值
    在th_begin、th_end中寻找合适的阈值
    '''
    assert img_gray.ndim == 2, "must input a gary_img"
    max_g = 0
    suitable_th = 0
    for threshold in range(th_begin, th_end, th_step):  # 前闭后开区间
        bin_img = img_gray > threshold
        bin_img_inv = img_gray <= threshold
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if 0 == fore_pix:
            break
        if 0 == back_pix:
            continue
        w0 = float(fore_pix) / img_gray.size
        u0 = float(np.sum(img_gray * bin_img)) / fore_pix
        w1 = float(back_pix) / img_gray.size
        u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix
        # 类间方差公式
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        # print('threshold, g:', threshold, g)
        if g > max_g:
            max_g = g
            suitable_th = threshold
    return suitable_th


# def getMeanValue(image):
#     '''
#     :param image: images中某一层的二维图像
#     :return: 图像的平均灰度
#     '''
#     pixel_val_sum = 0.0
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             pixel_val_sum += image[i][j]
#     # print(pixel_val_sum)
#     return abs(pixel_val_sum / (image.shape[0] * image.shape[1]))  # 取均值的绝对值


def closeDemo(image, structure_size):
    '''
    闭操作：补全细小连接
    '''
    kenerl = cv2.getStructuringElement(cv2.MORPH_RECT, (structure_size, structure_size))  # 定义结构元素的形状和大小
    # dst_1 = cv2.dilate(image, kenerl)  # 先膨胀
    # dst = cv2.erode(dst_1, kenerl)  # 腐蚀操作
    dst = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kenerl)  # 闭操作,与分开使用膨胀和腐蚀效果相同
    return dst


def largestConnectComponent(bw_img):
    '''
    求bw_img中的最大联通区域
    :param bw_img: 二值图
    :return: 最大连通区域
    '''
    labeled_img, num = label(bw_img, connectivity=2, background=0, return_num=True)  # 取连通区域,connectivity=2表示8领域
    max_label = 0
    max_num = 0
    for i in range(1, num + 1):  # lable的编号从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)  # 只留下最大连通区域，其他区域置0
    return lcc


def segCTGaryMatter(image, threshold_min=300, threshold_max=1000, threshold_step=2, binary_threshold=None):
    '''
    :param image:   二维的image
    :param threshold_min:  OTSU的最小阈值300
    :param threshold_max:  OTSU的最大阈值1000
    :param filter_size:   滤波器的大小
    :return:  分割好的二维图像,otsu阈值
    '''
    if binary_threshold is None:
        binary_threshold = otsuEnhance(image, threshold_min, threshold_max, threshold_step)  # 使用类间方差来计算合适的阈值
    ret, binary = cv2.threshold(image, binary_threshold, 255, cv2.THRESH_BINARY)  # 使用OTSU二值化

    image_max_region = largestConnectComponent(binary)  # 求img[i]的最大联通区域

    # 类型转换才能使用findContours函数

    # 将bool类型先转为np.uint8才能使用findContours
    image_max_region = image_max_region.astype(np.uint8)

    # 求外轮廓，避免中间的孔洞被保留,这里只求外轮廓
    _, contours, hierarchy = cv2.findContours(image_max_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    new_image = np.zeros(image.shape, np.uint8) * 255  # 创建与原图像等大的黑色幕布
    cv2.drawContours(new_image, contours, -1, (255, 255, 255), 3)  # contours是轮廓，-1表示全画，然后是轮廓线使用白色，厚度3
    new_image_fill = np.copy(new_image)

    cv2.fillPoly(new_image_fill, contours, (255, 255, 255))  # 填充外轮廓孔洞


    return new_image_fill, binary_threshold


def findSliceWidth(image, center_point):
    '''
    :param image:  分割脑部组织区域的图像
    :param centerW: 中点：取图像的中点
    :return: 返回边界
    '''

    H, W = np.where(image > 0)  # 都是整数
    if len(W) != 0:  #  当分割出来的区域有值才做处理
        # 使用4个数组用于存放左右两部分的点的坐标
        left_h = []
        left_w = []
        right_h = []
        right_w = []


        for p in range(len(W)):
            if W[p] <= center_point[1]:  # 中点偏左的坐标值放到left_h、left_w中
                left_h.append(H[p])
                left_w.append(W[p])
            else:  # 中点以下的点放到bottom_row、bottom_col中
                right_h.append(H[p])
                right_w.append(W[p])

        # 转成np.array类型
        left_h = np.array(left_h)
        left_w = np.array(left_w)
        right_h = np.array(right_h)
        right_w = np.array(right_w)

        # 左部分从中心点反向遍历
        l = center_point[1] - 1  # 索引从0开始,中点
        # left_cur_point = [left_h[w], left_w[w]]  # cur_point
        while l - 1 >= 0:
            left_cur_point = [left_h[l], left_w[l]]
            left_next_point = [left_h[l- 1], left_w[l - 1]]  # pre_point
            # 当前点的像素值大于0,下一个点的像素值等于0
            if image[left_next_point[0],left_next_point[1]] == 0 and image[left_cur_point[0], left_cur_point[1]] > 0:
                break
            l -= 1
            # if get_neighbors_sum(new_image_ fill, top_pre_point[0], top_pre_point[1]) < 1 and new_image_fill[
            #     top_pre_point[0], top_pre_point[1]] == 0:
            #     # if new_image_fill[pos_top1[0], pos_top1[1]] == 255 and new_image_fill[pos_top2[0], pos_top2[1]] == 0:
            #     if top_cur_point[0] <= min(row_img):  # 并且当前这个点已经等于分割交集的上端点,遍历结束可以停止搜索



        # 右部分从中心点沿着右方向正向遍历
        right_cur_point = [right_h[0], right_w[0]]  # 中点下一个位置
        for r in range(len(right_w[0])):
            right_cur_point = [right_h[r], right_w[r]]
            right_next_point = [right_h[r + 1], right_w[r + 1]]
            if image[right_cur_point[0], right_cur_point[1]] > 0 and image[
                right_next_point[0], right_next_point[1]] == 0:
                break  # 上一个点是白，下一个是黑，停止查找

        slice_width = l - r

        return slice_width

if __name__ == '__main__':
    CT_RotateMhdDir = r'C:\Users\Vicky\Desktop\03_26_images\CT_predict_30_result_check'  # 之前预测的30个CT_mhd,去除检查效果不好的几个数据
    CT_RotateMhdPaths = glob(CT_RotateMhdDir + '\\*.mhd')
    for mhdPaths in CT_RotateMhdPaths:
        imageArray, _, _ = readMhd(mhdPaths)  # 坐标顺序Z,H,W
        for z in range(imageArray.shape[0]):
            # 对旋转后的图像分割

            center_point = findCenterPoints(imageArray[z])  # 理想条件下,旋转之后的中心,图像的中心
            seg_CT_image = segCTGaryMatter(imageArray[z])
            cur_slice_width = findSliceWidth(seg_CT_image, center_point)  # 计算层上的长度,只用计算列的边界的差
