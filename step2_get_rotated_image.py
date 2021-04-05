import os
import time
from glob import glob
import numpy as np
import SimpleITK as sitk
import scipy
import scipy.ndimage
import skimage.transform as transform
from skimage.measure import label
import cv2
import json
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


def saveMhd(imageArray, origin, spacing, savePath):
    '''
    根据图像Array、origin、spacing构造itk image对象，并将其保存到savePath
    '''
    itkImage = sitk.GetImageFromArray(imageArray, isVector=False)
    itkImage.SetSpacing(spacing)
    itkImage.SetOrigin(origin)
    sitk.WriteImage(itkImage, savePath)


def getRotateMatrix3D(vectorZYX):
    '''
    获取将vectorZYX向量旋转到和x轴平行的变换矩阵（旋转之后，vectorZYX不一定指向x轴正方向，也可能方向相反，按最小角度旋转）
    vectorZYX 被旋转的向量，现在只支持三维
    返回旋转矩阵
    '''
    vectorZYX = np.array(vectorZYX, dtype=np.float)  # 默认当做列向量处理
    length = np.sqrt(np.matmul(vectorZYX.T, vectorZYX))  # 计算长度
    vectorZYX = vectorZYX / length  # 化为单位向量
    xAxis = np.array([0, 0, 1], dtype=np.float)  # x轴正方向的单位向量

    # 计算向量和x轴正方向的夹角的余弦值
    vTx = np.matmul(vectorZYX.T, xAxis)
    if vTx < 0:
        # 如果vectorZYX和x轴正方向夹角大于90度，那么后面求的是对于-vectorZYX向量的旋转矩阵
        vectorZYX = -vectorZYX
        vTx = -vTx

    cosTheta = vTx  # 由于两个向量都是单位向量，这里不需要除以模长，直接得到cos(\theta)
    sinTheta = np.sqrt(1.0 - cosTheta ** 2)  # 计算sin(\theta)
    k = np.cross(vectorZYX, xAxis)  # vectorZYX和x轴正方向单位向量的叉乘
    k = k / np.sqrt(np.matmul(k.T, k))  # 归一化为单位向量
    # 根据Rodrigues旋转公式可以获取旋转矩阵
    rotateMatrix = cosTheta * np.eye(3) + sinTheta * np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ], dtype=np.float32)

    return rotateMatrix


# def angleCorrect(image, mask, normalZYX, centerZYX, spacingZYX=None, targetSpacingZYX=None, spacingBack=True):
def angleCorrect(image, normalZYX, centerZYX, spacingZYX=None, targetSpacingZYX=None, spacingBack=True):
    '''
    按照矢状面法向量以及矢状面脑部中心点，对三维图像进行调整，使脑部图像居中且在xy层面对称
    image: 需要进行调整的图像，本函数不改变image的值，而是返回一个新的image，其形状和原始image相同
    normalZYX: 矢状面法向量，[w_z, w_y, w_x]，经过调整之后，该向量与x轴平行
    centerZYX: 矢状面中心点，经过调整之后，该点将位于整个图像的中心
    spacingZYX: 原始图像的spacing信息
    targetSpacingZYX: 进行旋转过程时，需要调整到哪个spacing下，如果为None则不调整，建议指定为[1, 1, 1]
    spacingBack: 如果指定了targetSpacingZYX，最终是否将spacing调整回来
    返回变换后的图像以及齐次坐标变换矩阵
    '''
    normalZYX = np.array(normalZYX, dtype=np.float32)
    centerZYX = np.array(centerZYX, dtype=np.float32)

    imageShapeZYX = np.array(image.shape, dtype=np.int)

    if targetSpacingZYX is not None:
        assert spacingZYX is not None, 'need spacingZYX to change spacing'

        targetSpacingZYX = [targetSpacingZYX[idx] if targetSpacingZYX[idx] is not None else spacingZYX[idx] for idx in
                            range(3)]  # 如果其中某项为None，则表示该维度不进行spacing转换，维持原spacing不变

        spacingZYX = np.array(spacingZYX, dtype=np.float32)
        targetSpacingZYX = np.array(targetSpacingZYX, dtype=np.float32)

        # 首先插值，将图片spacing换到targetSpacing, 法向量和中心点坐标也要做相应变换
        scaleZYX = spacingZYX / targetSpacingZYX

        _normalZYX = normalZYX / scaleZYX

        _centerZYX = centerZYX * scaleZYX

        scaleMatrix = np.array([
            [scaleZYX[0], 0, 0, 0],
            [0, scaleZYX[1], 0, 0],
            [0, 0, scaleZYX[2], 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        # 缩放后的图像大小发生改变
        newImageShapeZYX = imageShapeZYX.copy().astype(np.float32)
        newImageShapeZYX *= scaleZYX
        newImageShapeZYX = np.round(newImageShapeZYX).astype(np.int)
    else:
        scaleMatrix = np.eye(4, dtype=np.float32)
        _normalZYX = normalZYX.copy()
        _centerZYX = centerZYX.copy()
        newImageShapeZYX = imageShapeZYX.copy()

    # 将中心点移到原点位置
    translateMatrix = np.array([
        [1, 0, 0, -_centerZYX[0]],
        [0, 1, 0, -_centerZYX[1]],
        [0, 0, 1, -_centerZYX[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    # 计算旋转矩阵
    rotateMatrix = getRotateMatrix3D(_normalZYX)
    # 将旋转矩阵扩展为齐次形式
    rotateMatrix = np.pad(rotateMatrix, [(0, 1), (0, 1)], mode='constant', constant_values=0)
    rotateMatrix[3][3] = 1
    # 旋转之后需要将原点位置平移回图像中心
    deTranslateMatrix = np.array([
        [1, 0, 0, newImageShapeZYX[0] / 2.0],
        [0, 1, 0, newImageShapeZYX[1] / 2.0],
        [0, 0, 1, newImageShapeZYX[2] / 2.0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    if spacingBack and targetSpacingZYX is not None:
        # 如果有targetSpacingZYX，则需要把形状调回来
        deScaleZYX = 1.0 / scaleZYX
        deScaleMatrix = np.array([
            [deScaleZYX[0], 0, 0, 0],
            [0, deScaleZYX[1], 0, 0],
            [0, 0, deScaleZYX[2], 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        newImageShapeZYX = imageShapeZYX.copy()
    else:
        deScaleMatrix = np.eye(4, dtype=np.float32)

    # 完整的变换矩阵: deScaleMatrix * deTranslateMatrix * rotateMatrix * translateMatrix * scaleMatrix
    transformMatrix = deScaleMatrix.dot(deTranslateMatrix.dot(rotateMatrix.dot(translateMatrix.dot(scaleMatrix))))
    # 获取逆变换矩阵
    deTransformMatrix = np.linalg.inv(transformMatrix)

    # 构造变换之后的坐标网格，如果没有targetSpacing或者有spacingBack则和原图大小一致，否则需要一定的scale
    gridZ, gridY, gridX = np.mgrid[range(newImageShapeZYX[0]), range(newImageShapeZYX[1]), range(newImageShapeZYX[2])]
    # 拓展为齐次坐标
    grid1 = np.ones(newImageShapeZYX)  # (Z, H, W)
    gridZYX1 = np.stack([gridZ, gridY, gridX, grid1], axis=-1)

    # 进行映射和相应的插值
    # 将需要计算的坐标映射回原图相应位置：
    deTransformMatrix = deTransformMatrix[:3, :]  # 这里的映射矩阵不再需要最后一行，以减少计算量
    deTransformMatrix = np.expand_dims(deTransformMatrix, axis=(0, 1, 2))  # (1, 1, 1, 3, 4)
    # 下面的(1, 1, 1, 3, 4) * (Z, H, W, 4, 1)矩阵乘法得到的是(Z, H, W, 3, 1)，切片之后得到(Z, H, W, 3)
    oriZYX = np.matmul(deTransformMatrix, np.expand_dims(gridZYX1, axis=-1))[:, :, :, :, 0]
    # 原图上的坐标网格
    points = (range(imageShapeZYX[0]), range(imageShapeZYX[1]), range(imageShapeZYX[2]))
    # 需要插值的点
    xi = (oriZYX[..., 0], oriZYX[..., 1], oriZYX[..., 2])
    # 进行插值，需要插值的地方如果超出图像外，则用最小值代替
    interpolatedImg = scipy.interpolate.interpn(points, image, xi, method='linear', bounds_error=False,
                                                fill_value=image.min())
    # mask = mask.astype(np.float32)
    # interpolatedMask = scipy.interpolate.interpn(points, mask, xi, method='nearest', bounds_error=False, fill_value=mask.min())
    # 插值之后会变成float类型，需要修改回int16
    interpolatedImg = np.round(interpolatedImg).astype(np.int16)

    # interpolatedMask = np.round(interpolatedMask).astype(np.uint8)

    # interpolatedMask  = np.round(interpolatedMask > 0.3).astype(np.uint8)

    # return interpolatedImg, interpolatedMask
    return interpolatedImg


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
    :param threshold_min:  OTSU的最小阈值
    :param threshold_max:  OTSU的最大阈值
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

        oriImg = curSlice.copy()

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
            cv2.imwrite(savePath.replace('.png', '_ori_{}.png'.format(sliceIdx)), (oriImg * 255).astype(np.uint8))
            cv2.imwrite(savePath.replace('.png', '_gaussian_{}.png'.format(sliceIdx)),
                        (curSlice * 255).astype(np.uint8))
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
    '''
    按照step_1预测的中平面信息，对图像和mask进行相应旋转，旋转结果如果通过最终的对称度检查，则保存
    step_2:旋转、自检
    '''
    imageDir = '/mnt/data3/brain_angle_correction/lianying_data/CT_mhd_new'  # 联影CT50套

    # 先用一个数据测试新增的代码
    # imageDir = '/mnt/data1/wx/head_angle_correct/test_data'
    # maskDir   = '' # 所有待旋转标注放到这个文件夹下,CT脑卒中的mask
    planeParamsDictPath = '/mnt/data1/wx/head_angle_correct/UNITED_IMAGING_DATA/CT_plane_params_result/plane_params.json'

    imageSaveDir = '/mnt/data1/wx/head_angle_correct/UNITED_IMAGING_DATA/step2_CT_rotate_result'  # 预测结果目录
    # maskSaveDir  = ''

    checkSaveDir = '/mnt/data1/wx/head_angle_correct/UNITED_IMAGING_DATA/step2_CT_result_check_0402'

    with open(planeParamsDictPath, 'r') as fp:
        planeParamsDict = json.load(fp)

    if not os.path.exists(imageSaveDir):
        print('makedir: ', imageSaveDir)
        os.makedirs(imageSaveDir)

    # if os.path.exists(maskSaveDir):
    # 	os.makedirs(maskSaveDir)

    imagePaths = glob(os.path.join(imageDir, '*.mhd'))

    print('总共找到', len(imagePaths), '个mhd文件')

    # 画散点图的参数
    angleErrors = []  # 和金标准之间的角度差异值
    meanErrors = []  # 左右信息的差异值

    for imagePath in imagePaths:
        mhdName = os.path.basename(imagePath)

        print('curMHDName: ', mhdName)  # 打印当前mhd文件序号

        w = planeParamsDict[mhdName]['w']
        # b = planeParamsDict[mhdName]['b']
        centerZYX = planeParamsDict[mhdName]['centerZYX']

        # maskPath = os.path.join(maskDir, mhdName.replace('.mhd', '_seg.mhd'))

        image, origin, spacingXYZ = readMhd(imagePath)  # 这里的spacing为，xyz顺序
        # mask, _, _ = readMhd(maskPath)

        # image = image.astype(np.int16)

        spacingZYX = list(reversed(spacingXYZ))

        startTime = time.time()

        # 暂时还没有CT上脑卒中的mask,省略参数mask
        interpolatedImg = angleCorrect(image, w, centerZYX, spacingZYX=spacingZYX,
                                       targetSpacingZYX=(1, 1, 1), spacingBack=True)

        endTime = time.time()

        print('姿态调整用时：', endTime - startTime)

        mhdID = mhdName.split('.')[0]  # mhdID

        checkSaveDir = os.path.join(checkSaveDir, mhdID)

        if not os.path.exists(checkSaveDir):  # 新建用于存放check过程的目录
            os.makedirs(checkSaveDir)

        checkSavePath = os.path.join(checkSaveDir, mhdName.replace('.mhd', '.png'))  # 将自检过程保存成png

        # interpolatedImg, interpolatedMask = angleCorrect(image, mask,w, centerZYX, spacingZYX=spacingZYX, targetSpacingZYX=(1, 1, 1), spacingBack=True)

        checkResult, meanError = checkRotateResult(interpolatedImg, size=128, pixValueErrorThreshold=0.3,
                                                   pixNumErrorThreshold=400, savePath=checkSavePath, checkLayerNum=7)

        angleError = 0
        angleErrors.append(angleError)
        meanErrors.append(meanError)

        # 暂时全部保存
        # 重新拼接文件夹(一个病例一个文件夹)

        mhdSaveDir = os.path.join(imageSaveDir, mhdID)

        if not os.path.exists(mhdSaveDir):
            os.makedirs(mhdSaveDir)

        imageSavePath = os.path.join(mhdSaveDir, mhdName)

        print('mhdpath: ', imageSavePath)

        saveMhd(interpolatedImg, origin, spacingXYZ, imageSavePath)

    plt.figure(figsize=(10, 10), dpi=100)  # 设置画布大小，像素
    plt.title('meanErrors: {:.2f}'.format(np.mean(meanErrors)))

    # 联影的数据没有数据角度偏差,值为0,画出的图记录的就只有左右信息的偏差
    plt.scatter(angleErrors, meanErrors, label='result check')  # 画散点图并指定图片标签

    plt.legend()  # 显示图片中的标签
    plt.savefig('./{}.jpg'.format('step2_CT_rotate_scatter_0402'))  # 保存到项目的文件夹之下
    plt.close()
    # if checkResult:
    # 	imageSavePath = os.path.join(imageSaveDir, mhdName)
    # 	# maskSavePath  = os.path.join(maskSaveDir, mhdName.replace('.mhd', '_seg.mhd'))
    # 	saveMhd(interpolatedImg, origin, spacingXYZ, imageSavePath)
    # 	# saveMhd(interpolatedMask, origin, spacingXYZ, maskSavePath)
    # else:
    # 	print(imagePath, 'result check failed with error:', meanError)
