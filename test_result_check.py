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
    03_29:修改为对ZX面分割区域
	:param image:   二维的image
	:param threshold_min:  OTSU的最小阈值
	:param threshold_max:  OTSU的最大阈值
	:param filter_size:   滤波器的大小
	:return:  分割好的二维图像
	'''
    if binary_threshold is None:
        binary_threshold = OTSU_enhance(image, threshold_min, threshold_max, threshold_step)  # 使用类间方差来计算合适的阈值
    ret, binary = cv2.threshold(image, binary_threshold, 255, cv2.THRESH_BINARY)  # 使用OTSU二值化

    image = largestConnectComponent(binary)  # 求img[i]的最大联通区域

    # plt.figure()
    # plt.imshow

    return image, binary_threshold


def checkRotateResult(rotatedImage, size=128, pixValueErrorThreshold=0.3, pixNumErrorThreshold=400, savePath=None,
                      checkLayerNum=7):
    '''
	判断旋转后的图像是否是近似左右对称
	'''
    Z, H, W = rotatedImage.shape  # 现在读取出来的维度顺序

    # 计算冠状面的对称信息  Z,X,滚动条代表的是Y值

    # 计算需要检查的层
    # centerSliceIdx = Z // 2
    # sliceStart = max(centerSliceIdx - checkLayerNum // 2, 0)
    # sliceEnd = min(sliceStart + checkLayerNum, Z)  # 从以上层的起始位置再加7层

    # 对冠状面还是取中间几层,边缘信息会因各种原因不对称
    centerIdx = H // 2
    inxStart = max(centerIdx - checkLayerNum // 2, 0)
    inxEnd = min(inxStart + checkLayerNum, H)

    # 实际取出的层
    realHeight = inxEnd - inxStart

    print('H:  ', H, '     centerIdx:  ', centerIdx, '    inxStart:  ', inxStart, '    inxEnd:  ', inxEnd,
          '   realHeight:  ', realHeight)

    # 考虑到实际的层数可能和指定检查层数不同
    # realLayerNum = sliceEnd - sliceStart

    # if realLayerNum == 0:
    #     print('image is empty!!!!')
    #     return False

    if realHeight == 0:
        print('iamge height is zero!!!')
        return False

    meanError = 0
    binary_threshold = None
    for sliceIdx in range(inxStart, inxEnd):
        # 复制以防修改原图
        curSlice = rotatedImage[:, sliceIdx, :].copy()  # 传过去的是Z,X平面的二维图像

        curSlice, binary_threshold = get_gray_matter(curSlice, threshold_min=-500, threshold_max=-300, threshold_step=2,
                                                     binary_threshold=binary_threshold)
        curSlice = curSlice.astype(np.float32)

        # 缩放到固定大小
        curSlice = transform.resize(curSlice, (24, size), order=0, mode='constant', cval=0, anti_aliasing=False,
                                    preserve_range=True)
        # sigma 为 4 的高斯模糊
        curSlice = scipy.ndimage.filters.gaussian_filter(curSlice, 4, truncate=4.0)
        # 左右相减，求绝对值
        subMap = np.abs(curSlice - curSlice[:, ::-1])  # ::-1对X左右反转
        # print('左右相减: ', subMap)

        if savePath is not None:
            cv2.imwrite(savePath.replace('.png', '_{}.png'.format(sliceIdx)),
                        (curSlice * 255).astype(np.uint8))  # 保存的左右相减的结果
            cv2.imwrite(savePath.replace('.png', '_sub_{}.png'.format(sliceIdx)),
                        (subMap * 255).astype(np.uint8))  # 保存加上高斯模糊等的中间结果
        # 大于阈值的算作错误点，计算所有错误点个数
        meanError += np.sum(subMap > pixValueErrorThreshold)

    # 计算错误点个数平均值
    meanError /= realHeight

    print('realLayerNum:', realHeight, 'checked error:', meanError)

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

    # resultSaveDir = '/mnt/data3/brain_angle_correction/CT_result/'

    # resultDirName = 'no_rotate_line_masks_fixNoiseBug_CT_lianying_fixIMAReadbug'  # 'no_rotate_line_masks_gray_scale1_shift1_spacingBack_CT_lianying' #'no_rotate_line_masks_spacingBack_CT_lianying'

    # rotatedMHDDir = os.path.join(resultSaveDir, resultDirName)

    # resultSaveDir = '/mnt/data3/brain_angle_correction/CT_result/'
    # saveDir = os.path.join(resultSaveDir, resultDirName + '_result_check')  # 最终通过检查的才保存

    resultSaveDir = '/mnt/data1/wx/head_angle_correct/UNITED_IMAGING_DATA/step2_CT_predict_0402/'  # 检查结果和预测结果的外层路径

    resultDirName = '0009'

    rotatedMHDDir = os.path.join(resultSaveDir, resultDirName)

    saveDir = os.path.join(resultSaveDir, resultDirName + '_ZX_result_check')

    if not os.path.exists(saveDir):  # 路径不存在就先创建一个
        print('创建左右信息检查结果保存路径')
        os.makedirs(saveDir)
        print('路径保存成功')

    allMHDPaths = glob(os.path.join(rotatedMHDDir, '*.mhd'))  # 将以下目录中的mhd文件全部取出来

    # rotatedMHDDir = '/mnt/data3/brain_angle_correction/CT_result/no_rotate_line_masks_fixNoiseBug_CT_lianying_fixIMAReadbug/

    angleErrors = []
    meanErrors = []

    for MHDPath in allMHDPaths:
        MHDName = os.path.basename(MHDPath)  # 取目录中最后一个文件名——就是对应mhd文件名(包括mhd后缀)
        print('test_mhdname:  ', MHDName)
        # 文件中没有包含两个字符?预测完旋转后保存的时候把角度偏差一起保存的
        if '_e' not in MHDName:  # 联影的预测之后没有计算偏差可以计算
            angleError = 0
        else:
            angleError = float(MHDName.split('.mhd')[0].split('_e')[-1])  # 将角度偏差取出来

        imageArray, origin, spacing = readMhd(MHDPath)

        # 前面省略的返回值是是否检查通过的bool值
        _, meanError = checkRotateResult(imageArray, size=128, pixValueErrorThreshold=0.3, pixNumErrorThreshold=1000,
                                         savePath=os.path.join(saveDir, MHDName.replace('.mhd', '.png')),
                                         # 拼接文件路径和文件名(绝对路径)
                                         # 将旋转结果保存成png格式用于检查
                                         checkLayerNum=100)

        angleErrors.append(angleError)
        meanErrors.append(meanError)

    plt.figure(figsize=(10, 10), dpi=100)  # 设置画布大小，像素
    plt.title('meanErrors: {:.2f}'.format(np.mean(meanErrors)))

    # 联影的数据没有数据角度偏差,值为0,画出的图记录的就只有左右信息的偏差
    plt.scatter(angleErrors, meanErrors, label='result check')  # 画散点图并指定图片标签

    plt.legend()  # 显示图片中的标签
    plt.savefig('./{}.jpg'.format(resultDirName))  # 保存到项目的文件夹之下
    plt.close()
