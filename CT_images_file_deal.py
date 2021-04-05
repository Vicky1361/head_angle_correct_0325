import os
import shutil

# 将所有mhd全部放到一个文件夹下
path = '/mnt/data3/brain_angle_correction/lianying_data/CT_mhd'
new_path = '/mnt/data1/wx/head_angle_correct/UNITED_IMAGING_DATA/CT_MHD/'

count = 0
for root, dirs, files in os.walk(path, new_path):
    # print(root, dirs, files)

    for i in range(len(files)):
        count = count + 1
        # print(files[i])
        file_path = root + '/' + files[i]
        # print(file_path)
        new_file_path = new_path + files[i]
        # print(new_file_path)
        shutil.copy(file_path, new_file_path)

print(count)