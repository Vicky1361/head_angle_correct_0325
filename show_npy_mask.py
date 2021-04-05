import numpy as np
import cv2

# npyPath = r'/mnt/data1/wx/data/val_npy/line_masks/201_seg.npy'
npyPath = r'/mnt/data1/wx/data/train_npy/line_masks/0_seg.npy'

imArray = np.load(npyPath)

print(imArray.shape, imArray.dtype, imArray.min(), imArray.max())

imArray *= 255

cv2.imwrite('./test1.png', imArray[12])