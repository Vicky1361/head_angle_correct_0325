import os
import time
from glob import glob
import numpy as np
import torch
import SimpleITK as sitk
import skimage.transform as transform
from sklearn.cluster import KMeans
from sklearn import linear_model
import cv2
import json


def readMhd(mhdPath):
	'''
	根据文件路径读取mhd文件，并返回其内容Array、origin、spacing
	'''
	itkImage = sitk.ReadImage(mhdPath)
	imageArray = sitk.GetArrayFromImage(itkImage) #[Z, H, W]
	origin = itkImage.GetOrigin()
	spacing = itkImage.GetSpacing()
	return imageArray, origin, spacing


#  根据预测结果求中点
# def findCentrePoint(image, w, b):
# 	'''
# 	根据中平面寻找脑部中心点，中平面方程表示为：w[0] * z + w[1] * y + w[2] * x + b = 0
# 	w：平面参数，向量，list，[w_z, w_y, w_x]顺序
# 	b：平面参数，标量，float，表示偏置
# 	函数返回的中心点坐标为[z,y,x]顺序
# 	'''
# 	# 直接根据图像中心的z坐标和y坐标推算平面上对应点的x坐标，以此点作为中心点
# 	midZ = (image.shape[0] - 1) / 2.0
# 	midY = (image.shape[1] - 1) / 2.0
# 	if w[2] == 0:
# 		midX = (image.shape[2] - 1) / 2.0
# 	else:
# 		midX = (w[0] * midZ + w[1] * midY + b) / (-w[2])
# 	midPoint = [midZ, midY, midX]
# 	return midPoint


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
		image = np.pad(image, [(0, inputLayerNums - indexRange[1] + indexRange[0] - 1), (0, 0), (0, 0)], mode='constant', constant_values=0)
	else:
		# 如果层数过多则进行等间隔采样
		interval = ((indexRange[1] - indexRange[0] + 1) // inputLayerNums)
		indices = list(range((indexRange[1] - inputLayerNums * interval + 1), indexRange[1] + 1, interval))
		image = image[indices]

	# 归一化到0~1范围，并转为float32
	if normalizeRange is None:
		image = (image - image.min()).astype(np.float32) / (image.max() - image.min()).astype(np.float32)
	else:
		image = (np.clip(image, normalizeRange[0], normalizeRange[1]) - normalizeRange[0]).astype(np.float32) / (normalizeRange[1] - normalizeRange[0])

	# H,W归一化到512*512
	if image.shape[1] != 512 or image.shape[2] != 512:
		# 使用最邻近插值方式
		print('before', image.dtype, image.shape, image.min(), image.max())
		image = transform.resize(image, (inputLayerNums, 512, 512), order=0, mode='constant', cval=image.min(), anti_aliasing=True, preserve_range=True)
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
		x = x.unsqueeze(0) #(Z, H, W) -> (1, Z, H, W)
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

	# 计算脑部中心点的xy坐标
	yPredResized = yPred.copy()
	yPredResized = transform.resize(yPredResized, image.shape, order=1, mode='constant', cval=0, anti_aliasing=True,
									preserve_range=True)
	totalZYXCoord = np.where(yPredResized > 0.5)  # 预测出来的mask像素值在0-1区间
	# 按照之前计算出来的Z效果不好,现在用的预测线段上的x,y的平均值,Z = Z/2
	centerZYX = [(image.shape[0] - 1) / 2, int(np.mean(totalZYXCoord[1])), int(np.mean(totalZYXCoord[2]))]


	points = []
	# 对每一层进行处理，i表示预测的层数，index表示在原图中的坐标
	validLayerNums = 0
	for i, index in enumerate(indices):
		# 每一层都有一个mask
		# 首先变回原始大小
		layerPred = transform.resize(yPred[i], (image.shape[1], image.shape[2]), order=1, mode='constant', cval=0, anti_aliasing=True, preserve_range=True)
		
		if predImageSaveDir is not None:
			# 保存预测结果
			saveImage = (np.round(layerPred * 255)).astype(np.uint8)
			cv2.imwrite(os.path.join(predImageSaveDir, '{}.png'.format(index)), saveImage)
		
		centerPoints = np.array(np.where(layerPred > 0.99)).transpose((1, 0)) # [(y1, x1), (y2, x2), ...]
		
		# print(i, len(centerPoints))
		
		if centerPoints.shape[0] < 100:
			# 如果该层预测出来的mask小于100个像素点大于0.99，则放弃这一层的点
			continue
		else:
			centerPoints = centerPoints[np.random.choice(centerPoints.shape[0], size=100, replace=False)] # 大于100个点时，随机取其中100个点，以免不同层之间点数差距太大

		for centerPoint in centerPoints:
			if spacingZYX is not None:
				points.append([index * spacingZYX[0], centerPoint[0] * spacingZYX[1], centerPoint[1]  * spacingZYX[2]]) # (z, y, x)顺序
			else:
				points.append([index, centerPoint[0], centerPoint[1]]) # (z, y, x)顺序
		validLayerNums += 1

	if validLayerNums < 8:
		print('符合要求的层数太少：', validLayerNums)
		return None, None

	points = np.array(points)
	# 考虑到一般的平面都是和zy平面接近平行，因此用x = w[0] * z + w[1] * y + b这种形式去拟合平面会好些
	X = points[:, :2]  # (z,y)坐标
	Y = points[:, 2]  # (x)坐标
	# 利用RANSAC,根据y, x坐标去回归z坐标,得到一个平面方程
	# 测试直接用线性回归方程,之前这里用的linear_model.RANSACRegressor()

	# linereg = linear_model.LinearRegression()
	#
	# linereg.fit(X, Y)
	# # 获取平面方程的系数， x = w[0] * z + w[1] * y + b
	# w = linereg.coef_  # w[0]是z的系数，w[1]是y的系数
	# b = linereg.intercept_

	# 利用RANSAC,根据y, x坐标去回归z坐标,得到一个平面方程
	linereg = linear_model.RANSACRegressor(linear_model.LinearRegression())
	linereg.fit(X, Y)
	# 获取平面方程的系数， x = w[0] * z + w[1] * y + b
	w = linereg.estimator_.coef_  # w[0]是z的系数，w[1]是y的系数
	b = linereg.estimator_.intercept_
	
	# 对参数进行调整，使得平面方程表示为：w[0] * z + w[1] * y + w[2] * x + b = 0
	w = [-w[0], -w[1], 1] # 同时这个也是法向量，zyx顺序
	b = -b
	if spacingZYX is not None:
		w = [wi * spacing for wi, spacing in zip(w, spacingZYX)]
	return w, b, centerZYX


if __name__ == '__main__':
	'''
	预测中平面并保存平面参数
	'''
	os.environ['CUDA_VISIBLE_DEVICES'] = '5'
	import unet_modify # 导入模型定义

	imageDir  = '/mnt/data3/brain_angle_correction/lianying_data/CT_mhd' # 所有待预测数据放到这个文件夹下
	modelPath = '/mnt/data1/wx/head_angle_correct/model_save/unet_modify_group.pkl'#'/mnt/data3/brain_angle_correction/model_save/no_rotate_line_masks/unet_modify_group.pkl' # '/mnt/data3/brain_angle_correction/model_save/no_rotate_line_masks_gray_scale1_shift1/unet_modify_group.pkl'#'/mnt/data1/wx/graduation_project/model/unet_modify_group.pkl' # '/mnt/data3/brain_angle_correction/model_save/no_rotate/unet_modify_group.pkl' # 模型文件路径 '/mnt/data1/wx/result_project/model/unet_modify_group.pkl'
	planeParamsDictSavePath = '/mnt/data1/wx/head_angle_correct/UNITED_IMAGING_DATA/CT_plane_params_result'


	targetSpacingZYX = (1, 1, 1) # 旋转的时候最好调整下Z轴的spacing，否则由于有些层厚很大，层数太少，很容易旋转到图像外面被截掉
	savePredImage = True

	# 构造、加载模型
	model = unet_modify.unet_modify_res_group()
	model.load_state_dict(torch.load(modelPath))
	model = model.cuda()
	# model.eval() # 这里加了eval之后会导致预测效果较差，可能是由于输入数据的channel代表的含义每个mhd都不一致，导致BN如果固定参数无法适应

	imagePaths = glob(os.path.join(imageDir, '*.mhd'))

	print('总共找到', len(imagePaths), '个mhd文件')

	if not os.path.exists(planeParamsDictSavePath):
		os.makedirs(planeParamsDictSavePath)

	planeParamsDict = {}

	for imagePath in imagePaths:

		mhdName = os.path.basename(imagePath)

		image, origin, spacingXYZ = readMhd(imagePath)

		image = image.astype(np.int16)

		spacingZYX = list(reversed(spacingXYZ))  # 直接转有没有问题？

		if savePredImage:
			predImageSaveDir = os.path.join(planeParamsDictSavePath, 'predImage', mhdName.split('.mhd')[0])
		else:
			predImageSaveDir = None

		w, b, centerZYX = getCentralPlane(image, model, inputLayerNums=12, cuda=True, predImageSaveDir=predImageSaveDir)

		if w is None:
			print(imagePath, '预测中平面参数失败, 跳过该图像')
			continue
		else:
			# 获取中心点坐标
			# centerZYX = findCentrePoint(image, w, b)

			planeParamsDict[mhdName] = {
				'w': w,  # w_0 * z + w_1 * y + w_2 * x + b = 0 表示中平面
				'b': b,
				'centerZYX': centerZYX
			}

			with open(planeParamsDictSavePath + '/plane_params.json', 'w') as fp:
				json.dump(planeParamsDict, fp)
