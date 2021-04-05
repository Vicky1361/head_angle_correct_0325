import os
import shutil
from glob import glob
import numpy as np
import pydicom
import SimpleITK as sitk


def readIMA(imaDir):
    allImaPaths = glob(os.path.join(imaDir, '*.IMA'))
    if len(allImaPaths) == 0:
        allImaPaths = glob(os.path.join(imaDir, '*.dcm'))
    imaSliceNums  = len(allImaPaths)
    imageArray = None
    spacingZYX = None
    for imaPath in allImaPaths:
        imaFile = pydicom.read_file(imaPath)
        # HU = pixel_val * slope + intercept
        sliceImage = np.round(imaFile.pixel_array.astype(np.float32) * float(imaFile.RescaleSlope)).astype(np.int16) + int(np.round(float(imaFile.RescaleIntercept))) # 转换为HU值
        if imageArray is None:
            imageArray = np.zeros((imaSliceNums, sliceImage.shape[0], sliceImage.shape[1]), dtype=np.int16)
            spacingZYX = [float(imaFile.SliceThickness), imaFile.PixelSpacing[0], imaFile.PixelSpacing[1]]
        sliceIdx = int(imaFile.InstanceNumber) - 1
        imageArray[sliceIdx] = sliceImage
    return imageArray, spacingZYX


def ima2mhd(oriDataDir, targetDataDir, CT=True):
    if not os.path.exists(targetDataDir):
        os.makedirs(targetDataDir)
    print('targetDataDir')
    print(os.listdir(oriDataDir))
    patientNames = [patientName for patientName in os.listdir(oriDataDir)]
    for patientName in patientNames:
        if CT:
            sub_dir=os.path.join(oriDataDir, patientName, 'CT')
            if not os.path.exists(sub_dir):
                print('sub_dir not exists for ', sub_dir)
                sub_dir=os.path.join(oriDataDir, patientName, 'CT2') # 0046有CT1和CT2两个文件夹，其中CT1是两个SeriesNumber，不知道是什么意思，所以这里读取CT2
                if not os.path.exists(sub_dir):
                    continue
                imageArray, spacingZYX = readIMA(sub_dir)
                image = sitk.GetImageFromArray(imageArray)
                image.SetSpacing(list(reversed(spacingZYX)))
                sitk.WriteImage(image, os.path.join(targetDataDir, patientName + '.mhd'))
            else:
                imageArray, spacingZYX = readIMA(sub_dir)
                image = sitk.GetImageFromArray(imageArray)
                image.SetSpacing(list(reversed(spacingZYX)))
                sitk.WriteImage(image, os.path.join(targetDataDir, patientName + '.mhd'))
        else:
            mhd_dir = os.path.join(oriDataDir, patientName, 'MRI_MHD')
            if os.path.exists(mhd_dir):
                # 如果有mhd文件夹，直接复制
                shutil.copy(os.path.join(mhd_dir, patientName + '.mhd'), os.path.join(targetDataDir, patientName + '.mhd'))
                shutil.copy(os.path.join(mhd_dir, patientName + '.raw'), os.path.join(targetDataDir, patientName + '.raw'))
                continue
            sub_dir=os.path.join(oriDataDir, patientName, 'MRI')
            imageArray, spacingZYX = readIMA(sub_dir)
            image = sitk.GetImageFromArray(imageArray)
            image.SetSpacing(list(reversed(spacingZYX)))
            sitk.WriteImage(image, os.path.join(targetDataDir, patientName + '.mhd'))


if __name__  == '__main__':
    # oriDataDir=r'E:\brain_angle_correction\MRI_DATA\data\1-50'
    # targetDataDir=r'E:\brain_angle_correction\MRI_DATA\data_processed\1-50'
    # oriDataDir=r'E:\brain_angle_correction\MRI_DATA\data\HK_DATA10'
    # targetDataDir=r'E:\brain_angle_correction\MRI_DATA\data_processed\HK_DATA10'

    oriDataDir=r'/mnt/data3/brain_angle_correction/lianying_data/1-50'
    targetDataDir=r'/mnt/data3/brain_angle_correction/lianying_data/CT_mhd'

    # oriDataDir=r'/mnt/data3/brain_angle_correction/lianying_data/1-50'
    # targetDataDir=r'/mnt/data3/brain_angle_correction/lianying_data/MRI_mhd'

    print(targetDataDir)
    ima2mhd(oriDataDir, targetDataDir, CT=True)


