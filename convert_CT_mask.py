import os
import numpy as np
import glob
import SimpleITK as sitk
from sklearn import linear_model


# 读取mhd文件
def read_mhd(file_name):
    itkimage = sitk.ReadImage(file_name)
    image_array = sitk.GetArrayFromImage(itkimage)
    origin = itkimage.GetOrigin()
    spacing = itkimage.GetSpacing()
    return image_array, origin, spacing


def saveMhd(imageArray, origin, spacing, savePath):
    '''
    根据图像Array、origin、spacing构造itk image对象，并将其保存到savePath
    '''
    itkImage = sitk.GetImageFromArray(imageArray, isVector=False)
    itkImage.SetSpacing(spacing)
    itkImage.SetOrigin(origin)
    sitk.WriteImage(itkImage, savePath)


def drawLine(image, z, k, b, value, scale=1):
    coordY = np.array(range(int(np.round(image.shape[1] * scale))))
    coordY = coordY.astype(np.float32)
    coordY /= scale    
    coordX = k[0] * coordY + b

    coordY = np.round(coordY).astype(np.int)
    coordX = np.round(coordX).astype(np.int)

    validMask = np.logical_and(coordX >= 0, coordX < image.shape[2])
    validMask = np.logical_and(validMask, np.logical_and(coordY >= 0, coordY < image.shape[1]))
    
    coordY = coordY[validMask]
    coordX = coordX[validMask]
    image[z, coordY, coordX] = value


def drawLine2(image, z, k, b, value, scale=1):
    coordX = np.array(range(int(np.round(image.shape[2]))))
    coordX = coordX.astype(np.float32)
    coordX /= scale
    
    coordY = k[0] * coordX + b
    coordY = np.round(coordY).astype(np.int)    
    coordX = np.round(coordX).astype(np.int)
    
    validMask = np.logical_and(coordX >= 0, coordX < image.shape[2])
    validMask = np.logical_and(validMask, np.logical_and(coordY >= 0, coordY < image.shape[1]))
    
    coordY = coordY[validMask]
    coordX = coordX[validMask]
    image[z, coordY, coordX] = value


# mask_dir = r'/mnt/data1/wx/data/train/masks'
# target_dir = r'/mnt/data1/wx/data/train/line_masks'

mask_dir = r'/mnt/data1/wx/data/val/masks'
target_dir = r'/mnt/data1/wx/data/val/line_masks'

mask_path = glob.glob(os.path.join(mask_dir, '*.mhd'))

count = 0
result = {}

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for path in mask_path:
    mask, mask_origin, mask_spacing = read_mhd(path)
    new_mask = np.zeros(mask.shape, dtype=np.uint8)

    # 逐层拟合
    for i in range(mask.shape[0]):
        points = []
        coordinates = np.array(np.where(mask[i] > 0)).transpose((1, 0))
        if len(coordinates) > 50:
            coordinates = coordinates[np.random.choice(coordinates.shape[0], size=50, replace=False)]
        for coordinate in coordinates:
            points.append([coordinate[0], coordinate[1]])
        points = np.array(points)

        if (len(points.shape) < 2):
            continue

        X = points[:, 0].reshape(-1, 1)
        Y = points[:, 1]
        linereg = linear_model.LinearRegression()  # linear_model.RANSACRegressor() # 这里不需要用RANSAC算法
        linereg.fit(X, Y)

        b = linereg.intercept_  # linereg.estimator_.intercept_
        k = linereg.coef_  # linereg.estimator_.coef_

        if k > 1 or k < -1:
            X = points[:, 1].reshape(-1, 1)
            Y = points[:, 0]
            linereg = linear_model.LinearRegression()  # linear_model.RANSACRegressor() # 这里不需要用RANSAC算法
            linereg.fit(X, Y)
            b = linereg.intercept_  # linereg.estimator_.intercept_
            k = linereg.coef_  # linereg.estimator_.coef_
            drawLine2(new_mask, i, k, b, 1)
        else:
            drawLine(new_mask, i, k, b, 1)

    maskSavePath = os.path.join(target_dir, os.path.basename(path))

    saveMhd(new_mask, mask_origin, mask_spacing, maskSavePath)

    print(maskSavePath)
