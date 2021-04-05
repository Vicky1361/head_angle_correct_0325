import dataset_3d
import unet_modify
import dice_bce_loss
import time

from torch.utils import data
import torch
import glob
import os
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# 训练集images使用的还是原脑偏角项目的数据677个,masks是截断拟合直线的seg_line_mask,验证集也是原先的92个
train_img_dir = '/mnt/data1/wx/data/train_npy/images'  # 训练集原图的npy格式
train_mask_dir = '/mnt/data1/wx/data/train_npy/seg_line_masks'  # 训练集的mask截断拟合直线的npy格式的mask
val_img_dir = '/mnt/data1/wx/data/val_npy/images'  # 验证集原图像的npy格式
val_mask_dir = '/mnt/data1/wx/data/val_npy/seg_line_masks'  # 验证集截断直线的mask处理的npy格式
model_save_dir = '/mnt/data1/wx/head_angle_correct/model_save_test/'  # 模型保存的路径

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

train_img_paths = glob.glob(train_img_dir + '/*.npy')
train_mask_paths = train_img_paths.copy()
val_img_paths = glob.glob(val_img_dir + '/*.npy')
val_mask_paths = val_img_paths.copy()

for i in range(len(train_mask_paths)):
    train_mask_paths[i] = train_mask_paths[i].replace(train_img_dir, train_mask_dir)
    train_mask_paths[i] = train_mask_paths[i].replace('.npy', '_seg.npy')
for i in range(len(val_mask_paths)):
    val_mask_paths[i] = val_mask_paths[i].replace(val_img_dir, val_mask_dir)
    val_mask_paths[i] = val_mask_paths[i].replace('.npy', '_seg.npy')

batch_size = 16
model = unet_modify.unet_modify_res_group(input_channels=12, output_channels=12).cuda()
# model.load_state_dict(torch.load(os.path.join(model_save_dir, 'unet_modify_group.pkl')))
# model.load_state_dict(torch.load('/mnt/data1/wx/graduation_project/model/unet_modify_group.pkl'))
loss_fn = dice_bce_loss.my_loss().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
# scheduler = MultiStepLR(optimizer=optimizer, milestones=[10,20], gamma=0.2)

min_val_loss = 100
min_dice_coef = 100
total_val_loss = []
total_train_loss = []
for epoch in range(1):
    #  记录每个epoch开始的时间
    time_start = time.time()
    # 这里将训练时对数据集的归一化到12层试试,因为脑偏角的大部分数据都是大于24层,可能会导致之后本应补0层被预测了
    train_loader = data.DataLoader(
        dataset_3d.my_dataset_seg(train_img_paths, train_mask_paths, 1, is_crop=False, is_rotate=False), num_workers=4,
        drop_last=True, batch_size=batch_size, shuffle=True)
    train_loss_sum = 0
    train_dice_coef_sum = 0
    # scheduler.step()
    for i, (x, y) in enumerate(train_loader):
        # print(y.sum())
        x = Variable(x).cuda()
        y_true = Variable(y).cuda()

        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        train_loss_sum += loss.item()
        single_dice_coef = dice_bce_loss.dice_coef(y_true, y_pred).item()
        train_dice_coef_sum += single_dice_coef
        optimizer.step()
        if i % 2 == 0:
            print(
                'batch num:{}, loss: {}, dice_coef: {}, min_true: {}, max_true: {}, min_pred: {}, max_pred: {}'.format(
                    i, loss.item(), single_dice_coef, torch.min(y_true), torch.max(y_true), torch.min(y_pred),
                    torch.max(y_pred)))
    print('------- epoch: {}, epoch loss: {}, epoch dice: {} --------'.format(epoch, train_loss_sum / len(train_loader),
                                                                              train_dice_coef_sum / len(train_loader)))
    total_train_loss.append(train_loss_sum / len(train_loader))
    with torch.no_grad():
        # 由于联影的数据层数比较少,所以考虑将原数据处理成12层来训练和验证
        val_loader = data.DataLoader(
            dataset_3d.my_dataset_seg(val_img_paths, val_mask_paths, 12, is_crop=False, is_rotate=False), num_workers=4,
            batch_size=batch_size, shuffle=True)
        val_loss_sum = 0
        val_dice_coef_sum = 0
        for i, (x, y) in enumerate(val_loader):
            x = Variable(x).cuda()
            y_true = Variable(y).cuda()

            y_pred = model(x)
            val_loss_sum += loss_fn(y_pred, y_true).item()
            val_dice_coef_sum += dice_bce_loss.dice_coef(y_true, y_pred).item()
        print('-------- val: epoch: {}, epoch loss: {}, epoch dice_coef: {} ------'.format(epoch, val_loss_sum / len(
            val_loader), val_dice_coef_sum / len(val_loader)))
        total_val_loss.append(val_loss_sum / len(val_loader))
        if (val_loss_sum / len(val_loader)) < min_val_loss:
            min_dice_coef = val_dice_coef_sum / len(val_loader)
            print('----impove loss from {} to {}, dice_coef: {}--------'.format(min_val_loss,
                                                                                (val_loss_sum / len(val_loader)),
                                                                                min_dice_coef))
            min_val_loss = val_loss_sum / len(val_loader)  # 验证集的最小值一直在更新
            torch.save(model.state_dict(),
                       os.path.join(model_save_dir, 'unet_modify_group.pkl'))  # 现在还是根据loss小于最小值才更新模型参数
        else:
            print('----loss not impove from {}, dice_coef: {}'.format(min_val_loss, min_dice_coef))

    # 记录一下瞬时时间,画图的时间消耗不记录
    time_end = time.time()
    print('one epoch time cost :', time_end - time_start, 's')  # 训练一次的时间

    # plt.figure('loss')
    # plt.plot(range(0, epoch + 1), total_val_loss, label='val')
    # plt.plot(range(0, epoch + 1), total_train_loss, label='train')
    # plt.savefig(os.path.join(model_save_dir, 'unet_modify_group.jpg'))
    # plt.close('loss')


print('----- min_val_loss: {}'.format(min_val_loss))
