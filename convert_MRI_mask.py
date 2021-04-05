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


def drawPlane(image, w, b, value, scale=2):
	'''
	将平面：w[0] * z + w[1] * y + w[2] * x + b = 0画到三维图像中
	image: 图像数组, 会被修改
	w、b: 平面参数
	value: 用来表达平面的像素值
	scale：采样倍率，越大，所画出来的点越密集，防止由于离散点从而在平面中出现空洞
	无返回值，直接修改image
	'''
	imageShape = image.shape

	if w[0] != 0:
		coordY, coordX = np.mgrid[range(int(np.round(imageShape[1] * scale))), range(int(np.round(imageShape[2] * scale)))]
		coordY = coordY.astype(np.float32)
		coordX = coordX.astype(np.float32)
		coordY /= scale
		coordX /= scale
		coordZ = -(w[1] * coordY + w[2] * coordX + b) / w[0]
		
	elif w[1] != 0:
		coordZ, coordX = np.mgrid[range(int(np.round(imageShape[0] * scale))), range(int(np.round(imageShape[2] * scale)))]
		coordZ = coordZ.astype(np.float32)
		coordX = coordX.astype(np.float32)
		coordZ /= scale
		coordX /= scale
		coordY = -(w[0] * coordZ + w[2] * coordX + b) / w[1]
	elif w[2] != 0:
		coordZ, coordY = np.mgrid[range(int(np.round(imageShape[0] * scale))), range(int(np.round(imageShape[1] * scale)))]
		coordZ = coordZ.astype(np.float32)
		coordY = coordY.astype(np.float32)
		coordZ /= scale
		coordY /= scale
		coordX = -(w[0] * coordZ + w[1] * coordY + b) / w[2]
	else:
		assert False, "w中至少要有一个不为0"
	
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

	validIndices  = np.logical_and(validIndicesZ, np.logical_and(validIndicesY, validIndicesX))

	coordZ = coordZ[validIndices]
	coordY = coordY[validIndices]
	coordX = coordX[validIndices]

	image[coordZ, coordY, coordX] = value


mask_dir = r'E:\brain_angle_correction\MRI_DATA\data_processed\mask'
target_dir = r'E:\brain_angle_correction\MRI_DATA\data_processed\mask_convert'

mask_path = glob.glob(os.path.join(mask_dir, '*.mhd'))

count = 0
result = {}

if not os.path.exists(target_dir):
	os.makedirs(target_dir)

for path in mask_path:
	mask, mask_origin, mask_spacing = read_mhd(path)

	points = []
	for i in range(mask.shape[0]):
		coordinates = np.array(np.where(mask[i] > 0)).transpose((1, 0))
		if len(coordinates) > 50:
			coordinates = coordinates[np.random.choice(coordinates.shape[0], size=50, replace=False)]
		for coordinate in coordinates:
			points.append([i, coordinate[0], coordinate[1]])

	points = np.array(points)
	X = points[:, 1:]
	Y = points[:, 0]
	linereg = linear_model.LinearRegression() #linear_model.RANSACRegressor() # 这里不需要用RANSAC算法
	linereg.fit(X, Y)

	b = linereg.intercept_#linereg.estimator_.intercept_
	w = linereg.coef_ #linereg.estimator_.coef_

	# 将平面方程转换为：w[0] * z + w[1] * y + w[2] * x + b = 0
	normalZYX = [1, -w[0], -w[1]]
	b = -b

	new_mask = np.zeros(mask.shape, dtype=np.uint8)
	drawPlane(new_mask, normalZYX, b, 1, scale=4.0)
	
	maskSavePath = os.path.join(target_dir, os.path.basename(path))

	saveMhd(new_mask, mask_origin, mask_spacing, maskSavePath)

	print(maskSavePath, 'normalZYX', normalZYX, 'b', b)
