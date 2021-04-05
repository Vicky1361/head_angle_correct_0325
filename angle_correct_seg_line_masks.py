import os
import time
from glob import glob
import numpy as np
import torch
import SimpleITK as sitk
import scipy
import scipy.ndimage
import skimage.transform as transform
from sklearn.cluster import KMeans
from sklearn import linear_model
import cv2


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


def findCentrePoint(image, w, b):
    '''
    根据中平面寻找脑部中心点，中平面方程表示为：w[0] * z + w[1] * y + w[2] * x + b = 0
    w：平面参数，向量，list，[w_z, w_y, w_x]顺序
    b：平面参数，标量，float，表示偏置
    函数返回的中心点坐标为[z,y,x]顺序
    '''
    # 直接根据图像中心的z坐标和y坐标推算平面上对应点的x坐标，以此点作为中心点
    midZ = (image.shape[0] - 1) / 2.0
    midY = (image.shape[1] - 1) / 2.0
    if w[2] == 0:
        midX = (image.shape[2] - 1) / 2.0
    else:
        midX = (w[0] * midZ + w[1] * midY + b) / (-w[2])
    midPoint = [midZ, midY, midX]
    return midPoint


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


def angleCorrect(image, normalZYX, centerZYX, method='linear', spacingZYX=None, targetSpacingZYX=None,
                 spacingBack=False):
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
    interpolatedImg = scipy.interpolate.interpn(points, image, xi, method=method, bounds_error=False,
                                                fill_value=image.min())
    # 插值之后会变成float类型，需要修改回int16
    interpolatedImg = np.round(interpolatedImg).astype(np.int16)

    return interpolatedImg, transformMatrix


def split(scan):
    '''
    去除头部以外的层，返回剩下层的index list
    '''
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


def imagePreProcess(image, normalizeRange=None, inputLayerNums=24):
    '''
    对mhd图像进行split、采样、归一化、resize操作
    normalizeRange: 如果为None则按照最大值最小值归一化，否则应该提供一个归一化范围，list类型，第一个值为最小值，第二个值为最大值
    inputLayerNums: 采样层数，默认会将整个mhd数据采样24层做预测，不够24层则补零层
    '''
    # 如果层数大于50，则调用split函数将目标层数的index范围计算出来，表示为闭区间
    if image.shape[0] > 50:
        indexRange = split(image)
    else:
        indexRange = [0, image.shape[0] - 1]

    if (indexRange[1] - indexRange[0] + 1) <= inputLayerNums:
        # 如果层数不够，则拼接全零的层
        indices = [indexRange[0] + index for index in range(inputLayerNums)]
        image = np.pad(image, [(0, inputLayerNums - indexRange[1] + indexRange[0] - 1), (0, 0), (0, 0)],
                       mode='constant', constant_values=0)
    else:
        # 如果层数过多则进行等间隔采样
        interval = ((indexRange[1] - indexRange[0] + 1) // inputLayerNums)
        indices = list(range((indexRange[1] - inputLayerNums * interval + 1), indexRange[1] + 1, interval))
        image = image[indices]

    # 归一化到0~1范围，并转为float32
    if normalizeRange is None:
        image = (image - image.min()).astype(np.float32) / (image.max() - image.min()).astype(np.float32)
    else:
        image = (np.clip(image, normalizeRange[0], normalizeRange[1]) - normalizeRange[0]).astype(np.float32) / (
                    normalizeRange[1] - normalizeRange[0])

    # H,W归一化到512*512
    if image.shape[1] != 512 or image.shape[2] != 512:
        # 使用最邻近插值方式
        print('before', image.dtype, image.shape, image.min(), image.max())
        image = transform.resize(image, (inputLayerNums, 512, 512), order=0, mode='constant', cval=image.min(),
                                 anti_aliasing=True, preserve_range=True)
        print('after', image.dtype, image.shape, image.min(), image.max())

    return image, indices


def getCentralPlane(image, model, inputLayerNums=24, cuda=True, predImageSaveDir=None, spacingZYX=None):
    '''
    使用模型获取图像中平面参数
    image: 图像数组
    model: 用于预测的模型
    inputLayerNums: 送入模型之前的每个mhd采样层数
    cuda： 模型是否在GPU上，如果为True，需要保证传入的model已经在GPU上
    predImageSaveDir: 预测概率图保存的目录路径，如果为None则不保存
    spacingZYX: 是否在拟合之前，将三个方向的spacing统一，如果为None，则不进行统一，否则需要传入当前image的spacing信息
    '''
    # 复制原数组，以防对原数组有修改
    imageC = image.copy()

    imageC, indices = imagePreProcess(imageC, normalizeRange=(0, 300), inputLayerNums=inputLayerNums)

    # 默认会在model中前向传播的过程中保留中间结果以计算梯度，但是预测的时候是不需要梯度计算，因此使用no_grad以取消保留中间结果
    with torch.no_grad():
        # 将image转成pytorch tensor
        x = torch.from_numpy(imageC).to(torch.float)
        # 给数据增加batchsize维度
        x = x.unsqueeze(0)  # (Z, H, W) -> (1, Z, H, W)
        if cuda:
            x = x.cuda()
        yPred = model(x)
        # 将预测的mask去掉第一个1这个维度,1*inputLayerNums*512*512
        yPred = yPred.squeeze(0)
        # 将预测结果从cuda拿到cpu中,并且转成numpy模式
        if cuda:
            yPred = yPred.cpu().numpy()
        else:
            yPred = yPred.numpy()

    if predImageSaveDir is not None:
        if not os.path.exists(predImageSaveDir):
            os.makedirs(predImageSaveDir)

    points = []
    # 对每一层进行处理，i表示预测的层数，index表示在原图中的坐标
    validLayerNums = 0
    for i, index in enumerate(indices):
        # 每一层都有一个mask
        # 首先变回原始大小
        layerPred = transform.resize(yPred[i], (image.shape[1], image.shape[2]), order=1, mode='constant', cval=0,
                                     anti_aliasing=True, preserve_range=True)

        if predImageSaveDir is not None:
            # 保存预测结果
            saveImage = (np.round(layerPred * 255)).astype(np.uint8)
            cv2.imwrite(os.path.join(predImageSaveDir, '{}.png'.format(index)), saveImage)

        centerPoints = np.array(np.where(layerPred > 0.99)).transpose((1, 0))  # [(y1, x1), (y2, x2), ...]

        # print(i, len(centerPoints))

        if centerPoints.shape[0] < 100:
            # 如果该层预测出来的mask小于100个像素点大于0.99，则放弃这一层的点
            continue
        else:
            centerPoints = centerPoints[
                np.random.choice(centerPoints.shape[0], size=100, replace=False)]  # 大于100个点时，随机取其中100个点，以免不同层之间点数差距太大

        for centerPoint in centerPoints:
            if spacingZYX is not None:
                points.append([index * spacingZYX[0], centerPoint[0] * spacingZYX[1],
                               centerPoint[1] * spacingZYX[2]])  # (z, y, x)顺序
            else:
                points.append([index, centerPoint[0], centerPoint[1]])  # (z, y, x)顺序
        validLayerNums += 1

    if validLayerNums < 8:
        print('符合要求的层数太少：', validLayerNums)
        return None, None

    points = np.array(points)
    # 考虑到一般的平面都是和zy平面接近平行，因此用x = w[0] * z + w[1] * y + b这种形式去拟合平面会好些
    X = points[:, :2]  # (z,y)坐标
    Y = points[:, 2]  # (x)坐标
    # 利用RANSAC,根据y, x坐标去回归z坐标,得到一个平面方程
    linereg = linear_model.RANSACRegressor(linear_model.LinearRegression())
    # linear_model.LinearRegression
    linereg.fit(X, Y)
    # 获取平面方程的系数， x = w[0] * z + w[1] * y + b
    w = linereg.estimator_.coef_  # w[0]是z的系数，w[1]是y的系数
    b = linereg.estimator_.intercept_

    # 对参数进行调整，使得平面方程表示为：w[0] * z + w[1] * y + w[2] * x + b = 0
    w = [-w[0], -w[1], 1]  # 同时这个也是法向量，zyx顺序
    b = -b
    if spacingZYX is not None:
        w = [wi * spacing for wi, spacing in zip(w, spacingZYX)]
    return w, b


def drawPlane(image, w, b, value, scale=1):
    '''
    将平面：w[0] * z + w[1] * y + w[2] * x + b = 0画到三维图像中
    image: 图像数组, 会被修改
    w、b: 平面参数
    value: 用来表达平面的像素值
    scale：采样倍率，越大，所画出来的点越密集，防止由于离散点从而在平面中出现空洞
    无返回值，直接修改image
    '''
    imageShape = image.shape

    absW = np.abs(w)

    maxAxis = np.argmax(absW)

    if maxAxis == 0:
        coordY, coordX = np.mgrid[
            range(int(np.round(imageShape[1] * scale))), range(int(np.round(imageShape[2] * scale)))]
        coordY = coordY.astype(np.float32)
        coordX = coordX.astype(np.float32)
        coordY /= scale
        coordX /= scale
        coordZ = -(w[1] * coordY + w[2] * coordX + b) / w[0]
    elif maxAxis == 1:
        coordZ, coordX = np.mgrid[
            range(int(np.round(imageShape[0] * scale))), range(int(np.round(imageShape[2] * scale)))]
        coordZ = coordZ.astype(np.float32)
        coordX = coordX.astype(np.float32)
        coordZ /= scale
        coordX /= scale
        coordY = -(w[0] * coordZ + w[2] * coordX + b) / w[1]
    else:
        coordZ, coordY = np.mgrid[
            range(int(np.round(imageShape[0] * scale))), range(int(np.round(imageShape[1] * scale)))]
        coordZ = coordZ.astype(np.float32)
        coordY = coordY.astype(np.float32)
        coordZ /= scale
        coordY /= scale
        coordX = -(w[0] * coordZ + w[1] * coordY + b) / w[2]

    coordZ = np.round(coordZ).astype(np.int)
    coordY = np.round(coordY).astype(np.int)
    coordX = np.round(coordX).astype(np.int)

    coordZ = coordZ.reshape(-1)
    coordY = coordY.reshape(-1)
    coordX = coordX.reshape(-1)

    # 筛选合法的坐标位置
    validIndicesZ = np.logical_and(coordZ >= 0, coordZ < imageShape[0])
    validIndicesY = np.logical_and(coordY >= 0, coordY < imageShape[1])
    validIndicesX = np.logical_and(coordX >= 0, coordX < imageShape[2])

    validIndices = np.logical_and(validIndicesZ, np.logical_and(validIndicesY, validIndicesX))

    coordZ = coordZ[validIndices]
    coordY = coordY[validIndices]
    coordX = coordX[validIndices]

    image[coordZ, coordY, coordX] = value


def calVectorIncludeAngle(vector1, vector2):
    '''
    计算两个向量之间的夹角，返回角度
    '''
    # print(vector1, vector2)
    vector1 = np.array(vector1, dtype=np.float)  # 默认当做列向量处理
    length = np.sqrt(np.matmul(vector1.T, vector1))  # 计算长度
    vector1 = vector1 / length  # 化为单位向量

    vector2 = np.array(vector2, dtype=np.float)  # 默认当做列向量处理
    length = np.sqrt(np.matmul(vector2.T, vector2))  # 计算长度
    vector2 = vector2 / length  # 化为单位向量

    cosTheta = np.matmul(vector1.T, vector2)
    if cosTheta < 0:  # 两个向量有可能反向，这里进行处理
        cosTheta = -cosTheta
    theta = np.arccos(cosTheta)  # 这里返回的是弧度制单位

    return theta / np.pi * 180  # 转成角度返回


def saveRotatedImage(image, imageSavePath, w, b, centerZYX, spacingZYX, targetSpacingZYX=None, angleError=None,
                     drawPlaneValue=None, spacingBack=False):
    ''' 将mhd图像矩阵旋转、平移到正确的姿态然后保存
    image: mhd图像矩阵
    imageSavePath: 图像保存路径
    w: 已知中平面的法向量，中平面可以表示为：w[0] * z + w[1] * y + w[2] * x + b = 0
    b: 已知中平面的平面偏移参数
    centerZYX: 需要平移到图像中心的位置坐标，旋转操作也是绕该位置进行
    spacingZYX: image目前的spacing
    targetSpacingZYX: 需要调整image到目标spacing，如果为None，则不调整spacing
    angleError: 预测的角度偏差，如果传入，则会在保存mhd时，在名字中加入该项，主要用于debug
    drawPlaneValue: 把中平面画到mhd上去时所用的值，如果为None，则不画中平面
    '''

    if angleError is not None:
        imageSavePath = imageSavePath.replace('.mhd', '_e{:.2f}.mhd'.format(angleError))

    startTime = time.time()

    normalZYX = w

    if targetSpacingZYX is not None and not spacingBack:
        newSpacingXYZ = list(reversed(targetSpacingZYX))  # 这里不是很精确，因为变换之后的shape肯定是取整数的，如果后面要做精确操作，这里的新spacing不能这样直接赋值
    else:
        newSpacingXYZ = list(reversed(spacingZYX))

    # 进行图像旋转，返回值包括旋转后的图像以及齐次坐标变换矩阵
    rotatedImage, transformMatrix = angleCorrect(image, normalZYX, centerZYX, spacingZYX=spacingZYX,
                                                 targetSpacingZYX=targetSpacingZYX, spacingBack=spacingBack)

    print('save image:', imageSavePath, rotatedImage.shape, rotatedImage.max(), rotatedImage.min(), rotatedImage.dtype)

    if drawPlaneValue is not None:
        # 将平面画出来方便查看
        # 原来的平面方程系数(w_z, w_y, w_x, b)
        planeParams = np.array([normalZYX[i] for i in range(3)] + [b, ], dtype=np.float32)
        # 经过变换之后的平面方程系数
        newPlaneParams = (planeParams.T.dot(np.linalg.inv(transformMatrix))).T

        newPlaneParams /= np.max(np.abs(newPlaneParams))

        drawPlane(rotatedImage, newPlaneParams[:3], newPlaneParams[3], drawPlaneValue)

    endTime = time.time()
    print('姿态调整用时：', endTime - startTime)

    # 保存旋转之后的图像
    saveMhd(rotatedImage, origin, newSpacingXYZ, imageSavePath)


if __name__ == '__main__':
    '''
    中平面预测以及图像旋转完整流程代码
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    import unet_modify  # 导入模型定义

    # imageDir  = '/mnt/data1/wx/data/val/images' # 所有待测试数据放到这个文件夹下
    imageDir = '/mnt/data3/brain_angle_correction/lianying_data/CT_mhd'  # 所有待测试数据放到这个文件夹下
    modelPath = '/mnt/data1/wx/head_angle_correct/model_save/unet_modify_group.pkl'  # '/mnt/data3/brain_angle_correction/model_save/no_rotate_line_masks/unet_modify_group.pkl' # '/mnt/data3/brain_angle_correction/model_save/no_rotate_line_masks_gray_scale1_shift1/unet_modify_group.pkl'#'/mnt/data1/wx/graduation_project/model/unet_modify_group.pkl' # '/mnt/data3/brain_angle_correction/model_save/no_rotate/unet_modify_group.pkl' # 模型文件路径 '/mnt/data1/wx/result_project/model/unet_modify_group.pkl'
    # dstDir    = '/mnt/data3/brain_angle_correction/CT_result/no_rotate_line_masks_fixNoiseBug_allImage_spacingBack' # 完成处理之后，用于保存结果的文件夹
    dstDir = '/mnt/data1/wx/head_angle_correct/UNITED_IMAGING_DATA/CT_predict_50_3'
    normalGTPath = None  # '/mnt/data3/brain_angle_correction/GT_2_26.npy' # None #
    # normalGTPath  = './GT_3_16.npy' # None #
    showPlane = False
    targetSpacingZYX = (1, 1, 1)  # 旋转的时候最好调整下Z轴的spacing，否则由于有些层厚很大，层数太少，很容易旋转到图像外面被截掉
    savePredImage = True
    spacingBack = True

    if normalGTPath is not None:
        normalGTDict = np.load(normalGTPath, allow_pickle=True).item()
    else:
        normalGTDict = None

    # 构造、加载模型
    model = unet_modify.unet_modify_res_group()
    model.load_state_dict(torch.load(modelPath))
    model = model.cuda()
    # model.eval() # 这里加了eval之后会导致预测效果较差，可能是由于输入数据的channel代表的含义每个mhd都不一致，导致BN如果固定参数无法适应

    imagePaths = glob(os.path.join(imageDir, '*.mhd'))
    # imagePaths.sort()

    print('总共找到', len(imagePaths), '个mhd文件')

    if not os.path.exists(dstDir):
        os.makedirs(dstDir)

    angleErrors = []
    angleErrorsBeforeNormSpacing = []

    for imagePath in imagePaths:
        mhdName = os.path.basename(imagePath)

        image, origin, spacingXYZ = readMhd(imagePath)  # 这里的spacing为，xyz顺序

        # image = image[::-1, :, :]

        image = image.astype(np.int16)

        spacingZYX = list(reversed(spacingXYZ))

        if savePredImage:
            predImageSaveDir = os.path.join(dstDir, 'predImage', mhdName.split('.mhd')[0])
        else:
            predImageSaveDir = None

        w, b = getCentralPlane(image, model, inputLayerNums=12, cuda=True,
                               predImageSaveDir=predImageSaveDir)  # 这里的输出w就是法向量

        if w is None:
            print(imagePath, '预测中平面参数失败, 跳过该图像')
            continue

        # 获取中心点坐标
        centerZYX = findCentrePoint(image, w, b)

        imageSavePath = os.path.join(dstDir, os.path.basename(imagePath).replace('.mhd', '_rotated.mhd'))

        if normalGTDict is not None:
            # 读取ground truth计算出来的法向量，和预测出来的法向量计算角度，作为评估标准
            normalGT = normalGTDict[mhdName][:3]
            angleErrorBeforeNormSpacing = calVectorIncludeAngle(w,
                                                                normalGT)  # 这里计算的是spacing没有统一的角度差异，由于z轴spacing一般远大于x和y轴，因此微小的角度差异会被放大
            angleErrorsBeforeNormSpacing.append(angleErrorBeforeNormSpacing)

            normalGTNormSpacing = [normalGT[idx] / spacingZYX[idx] for idx in range(3)]
            wNormSpacing = [w[idx] / spacingZYX[idx] for idx in range(3)]
            angleError = calVectorIncludeAngle(wNormSpacing, normalGTNormSpacing)  # 这里计算的是spacing统一之后的角度差异
            angleErrors.append(angleError)
            print(mhdName, 'angle with gt:{:.2f}, angle with gt before norm spacing:{:.2f}'.format(angleError,
                                                                                                   angleErrorBeforeNormSpacing))

            # if angleError >= 2:
            saveRotatedImage(image, imageSavePath, w, b, centerZYX, spacingZYX, targetSpacingZYX=targetSpacingZYX,
                             angleError=angleError, drawPlaneValue=9999 if showPlane else None, spacingBack=spacingBack)
        else:
            saveRotatedImage(image, imageSavePath, w, b, centerZYX, spacingZYX, targetSpacingZYX=targetSpacingZYX,
                             angleError=None, drawPlaneValue=9999 if showPlane else None, spacingBack=spacingBack)

    if normalGTDict is not None:
        print('mean angle error: {:.2f}, mean angle error before norm spacing: {:.2f}'.format(np.mean(angleErrors),
                                                                                              np.mean(
                                                                                                  angleErrorsBeforeNormSpacing)))