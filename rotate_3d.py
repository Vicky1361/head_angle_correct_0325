import numpy as np
import math
import SimpleITK as sitk
import glob
import matplotlib.pyplot as plt

def get_coordinate(shape0, shape1, shape2):
    yz = shape1 * shape2
    coordinate_matrix = np.ones((4, shape0 * shape1 * shape2), dtype=np.uint16)
    for i in range(shape0):
        for j in range(shape1):
            for k in range(shape2):
                index = i * yz + j * shape2 + k
                coordinate_matrix[0][index] = i
                coordinate_matrix[1][index] = j
                coordinate_matrix[2][index] = k
    return coordinate_matrix

def get_translation_matrix(tx,ty,tz):
    translation_matrix = np.eye(4)
    translation_matrix[0][3] = tx
    translation_matrix[1][3] = ty
    translation_matrix[2][3] = tz
    return translation_matrix

def get_rotate_matrix(angle):
    rotate_matrix = np.eye(4)
    rotate_matrix[0][0] = math.cos(angle)
    rotate_matrix[0][1] = -math.sin(angle)
    rotate_matrix[1][0] = math.sin(angle)
    rotate_matrix[1][1] = math.cos(angle)
    return rotate_matrix

coordinate_array = get_coordinate(512, 512, 24)
translation_matrix_1 = get_translation_matrix(-256, -256, 0)
translation_matrix_2 = get_translation_matrix(256, 256, 0)

#自己实现的三维旋转
def rotate_img_seg(img, mask, angle):
    rotate_matrix = get_rotate_matrix(angle)
    new_coordinate_array = translation_matrix_2.dot(rotate_matrix.dot(translation_matrix_1.dot(coordinate_array)))
    new_coordinate_array = np.rint(new_coordinate_array).astype(int)
    new_coordinate_array[np.where(new_coordinate_array < 0)] = 0
    new_coordinate_array[0][np.where(new_coordinate_array[0] >= 512)] = 511
    new_coordinate_array[1][np.where(new_coordinate_array[1] >= 512)] = 511
    new_coordinate_array[2][np.where(new_coordinate_array[2] >= 24)] = 23

    new_img = np.zeros(img.shape, dtype=np.float32)
    new_mask = np.zeros(mask.shape, dtype=np.float32)
    new_img[coordinate_array[0, :], coordinate_array[1, :], coordinate_array[2, :]] = img[new_coordinate_array[0, :], new_coordinate_array[1, :], new_coordinate_array[2, :]]
    new_mask[coordinate_array[0, :], coordinate_array[1, :], coordinate_array[2, :]] = mask[new_coordinate_array[0, :], new_coordinate_array[1, :], new_coordinate_array[2, :]]
    return new_img, new_mask

if __name__ == '__main__':
    # test_path = '/mnt/data1/llx/MRSpine/data/bad_image/20150512095802_Image.mhd'
    # img_array = sitk.GetArrayFromImage(sitk.ReadImage(test_path))
    # print('llx')
    img_dir = '/mnt/data1/llx/head_angle/3d/val/images/'
    img_paths = glob.glob(img_dir + '*.mhd')
    for path in img_paths:
        img_array = sitk.GetArrayFromImage(sitk.ReadImage(path))
        img_array = 255.0 * (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
        img_array = img_array.transpose((1,2,0)).astype(np.float32)
        if img_array.shape[2] > 24:
            img_array = img_array[:, :, :24]
        angle = 0.1 * math.pi

        _, new_img_array, _ = rotate_img(img_array, img_array, img_array, angle)
        '''
        rotate_matrix = get_rotate_matrix(angle)
        new_coordinate_array = translation_matrix_2.dot(rotate_matrix.dot(translation_matrix_1.dot(coordinate_array)))
        new_coordinate_array = np.rint(new_coordinate_array).astype(int)
        new_coordinate_array[np.where(new_coordinate_array < 0)] = 0
        new_coordinate_array[0][np.where(new_coordinate_array[0] >= 24)] = 23
        new_coordinate_array[1][np.where(new_coordinate_array[1] >= 512)] = 511
        new_coordinate_array[2][np.where(new_coordinate_array[2] >= 512)] = 511

        new_img_array = np.zeros(img_array.shape, dtype=np.float16)
        new_img_array[coordinate_array[0, :], coordinate_array[1, :], coordinate_array[2, :]] = img_array[new_coordinate_array[0, :], new_coordinate_array[1, :], new_coordinate_array[2, :]]
        # new_img_array = 255*(new_img_array - np.min(new_img_array)) / (np.max(new_img_array) - np.min(new_img_array))
        '''
        new_img_array = new_img_array.astype(np.uint8)
        img_array = img_array.astype(np.uint8)
        for i in range(24):
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(img_array[:, :, i])
            plt.subplot(2, 1, 2)
            plt.imshow(new_img_array[:, :, i])
            plt.show()