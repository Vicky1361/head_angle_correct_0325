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

val_mask_dir = '/mnt/data1/wx/data/val/line_masks'

mask_path = glob.glob(os.path.join(val_mask_dir, '*.mhd'))

count = 0
result = {}

for path in mask_path:
	mask, mask_origin, mask_spacing = read_mhd(path)

	points = []
	for i in range(mask.shape[0]):
		coordinates = np.array(np.where(mask[i] > 0.95)).transpose((1, 0))
		for coordinate in coordinates:
			points.append([i, coordinate[0], coordinate[1]])

	points = np.array(points)
	X = points[:, 1:]
	Y = points[:, 0]
	linereg = linear_model.RANSACRegressor(linear_model.LinearRegression())
	linereg.fit(X, Y)

	b = linereg.estimator_.intercept_
	w = linereg.estimator_.coef_

	normal_vector = np.array([1, -w[0], -w[1], -b])  # (Z,Y,X)顺序的法向量
	file_name = os.path.basename(path)
	file_name = file_name.split('_')[0]+'.mhd'
	result[file_name] = normal_vector
	print(file_name, normal_vector)
	count += 1

np.save('GT_3_16.npy', result)
print(count)
