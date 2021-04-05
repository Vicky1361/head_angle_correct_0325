import os
import time
import numpy as np
from glob import glob
import SimpleITK as sitk
import scipy
import scipy.ndimage
import skimage.transform as transform
from skimage.measure import label
import cv2
import matplotlib.pyplot as plt


def readMhd(mhdPath):
    '''
    根据文件路径读取mhd文件，并返回其内容Array、origin、spacing
    '''
    itkImage = sitk.ReadImage(mhdPath)
    imageArray = sitk.GetArrayFromImage(itkImage)  # [Z, H, W]
    origin = itkImage.GetOrigin()
    spacing = itkImage.GetSpacing()
    return imageArray, origin, spacing


def OTSU_enhance(img_gray, th_begin, th_end, th_step=1):
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


def get_mean_value(image):
    '''
    :param image: images中某一层的二维图像
    :return: 图像的平均灰度
    '''
    pixel_val_sum = 0.0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_val_sum += image[i][j]
    # print(pixel_val_sum)
    return abs(pixel_val_sum / (image.shape[0] * image.shape[1]))  # 取均值的绝对值


def close_demo(image, structure_size):
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


def get_gray_matter(image, threshold_min=300, threshold_max=1000, threshold_step=2, binary_threshold=None):
    '''
    :param image:   二维的image
    :param threshold_min:  OTSU的最小阈值300
    :param threshold_max:  OTSU的最大阈值1000
    :param filter_size:   滤波器的大小
    :return:  分割好的二维图像
    '''
    if binary_threshold is None:
        binary_threshold = OTSU_enhance(image, threshold_min, threshold_max, threshold_step)  # 使用类间方差来计算合适的阈值
    ret, binary = cv2.threshold(image, binary_threshold, 255, cv2.THRESH_BINARY)  # 使用OTSU二值化

    image = largestConnectComponent(binary)  # 求img[i]的最大联通区域

    return image, binary_threshold


def checkRotateResult(rotatedImage, size=128, pixValueErrorThreshold=0.3, pixNumErrorThreshold=400, savePath=None,
                      checkLayerNum=7):
    '''
    判断旋转后的图像是否是近似左右对称
    '''
    Z, H, W = rotatedImage.shape

    # 计算需要检查的层
    centerSliceIdx = Z // 2
    sliceStart = max(centerSliceIdx - checkLayerNum // 2, 0)
    sliceEnd = min(sliceStart + checkLayerNum, Z)
    # 考虑到实际的层数可能和指定检查层数不同
    realLayerNum = sliceEnd - sliceStart

    if realLayerNum == 0:
        print('image is empty!!!!')
        return False

    meanError = 0
    binary_threshold = None
    for sliceIdx in range(sliceStart, sliceEnd):
        # 复制以防修改原图
        curSlice = rotatedImage[sliceIdx].copy()

        # 联影
        curSlice, binary_threshold = get_gray_matter(curSlice, threshold_min=-500, threshold_max=-300, threshold_step=2,
                                                     binary_threshold=binary_threshold)
        curSlice = curSlice.astype(np.float32)

        # 缩放到固定大小
        curSlice = transform.resize(curSlice, (size, size), order=0, mode='constant', cval=0, anti_aliasing=False,
                                    preserve_range=True)
        # sigma 为 4 的高斯模糊
        curSlice = scipy.ndimage.filters.gaussian_filter(curSlice, 4, truncate=4.0)
        # 左右相减，求绝对值
        subMap = np.abs(curSlice - curSlice[:, ::-1])

        if savePath is not None:
            cv2.imwrite(savePath.replace('.png', '_{}.png'.format(sliceIdx)), (curSlice * 255).astype(np.uint8))
            cv2.imwrite(savePath.replace('.png', '_sub_{}.png'.format(sliceIdx)), (subMap * 255).astype(np.uint8))
        # 大于阈值的算作错误点，计算所有错误点个数
        meanError += np.sum(subMap > pixValueErrorThreshold)
    # 计算错误点个数平均值
    meanError /= realLayerNum

    print('realLayerNum:', realLayerNum, 'checked error:', meanError)

    if meanError > pixNumErrorThreshold:
        # 错误点个数太多则检查不通过
        return False, meanError
    else:
        return True, meanError


if __name__ == '__main__':
    # rotatedMHDDir = '/mnt/data3/brain_angle_correction/CT_result/no_rotate_line_masks_gray_scale1_shift1_spacingBack_CT_lianying'#'/mnt/data3/brain_angle_correction/result_no_rotate_line_masks_gray_scale1_shift1_allImage_order1'
    # saveDir       = '/mnt/data3/brain_angle_correction/CT_result/no_rotate_line_masks_gray_scale1_shift1_spacingBack_CT_lianying_result_check'
    # rotatedMHDDir = '/mnt/data3/brain_angle_correction/CT_result/no_rotate_line_masks_gray_scale1_shift1_allImage_spacingBack'
    # saveDir       = '/mnt/data3/brain_angle_correction/CT_result/no_rotate_line_masks_gray_scale1_shift1_allImage_spacingBack_result_check'

    resultSaveDir = '/mnt/data1/wx/head_angle_correct/UNITED_IMAGING_DATA/'

    resultDirName = 'CT_predict_50_3'  # 'no_rotate_line_masks_gray_scale1_shift1_spacingBack_CT_lianying' #'no_rotate_line_masks_spacingBack_CT_lianying'

    rotatedMHDDir = os.path.join(resultSaveDir, resultDirName)

    saveDir = os.path.join(resultSaveDir, resultDirName + '_XY_result_check')

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    allMHDPaths = glob(os.path.join(rotatedMHDDir, '*.mhd'))  # 取出路径中的所有mhd文件

    angleErrors = []
    meanErrors = []

    for MHDPath in allMHDPaths:
        MHDName = os.path.basename(MHDPath)

        if '_e' not in MHDName:
            angleError = 0
        else:
            angleError = float(MHDName.split('.mhd')[0].split('_e')[-1])

        imageArray, origin, spacing = readMhd(MHDPath)

        print(MHDName)
        _, meanError = checkRotateResult(imageArray, size=128, pixValueErrorThreshold=0.3, pixNumErrorThreshold=400,
                                         savePath=os.path.join(saveDir, MHDName.replace('.mhd', '.png')),
                                         checkLayerNum=7)

        angleErrors.append(angleError)
        meanErrors.append(meanError)

    plt.figure(figsize=(10, 10), dpi=100)  # 设置画布大小，像素
    plt.title('meanErrors: {:.2f}'.format(np.mean(meanErrors)))
    plt.scatter(angleErrors, meanErrors, label='result check')  # 画散点图并指定图片标签
    plt.legend()  # 显示图片中的标签
    plt.savefig('./{}.jpg'.format(resultDirName + '_scatter '))  # 保存在项目路径下
