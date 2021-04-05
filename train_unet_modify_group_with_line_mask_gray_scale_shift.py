import dataset_3d
import unet_modify
import dice_bce_loss

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

os.environ['CUDA_VISIBLE_DEVICES'] = '9'

train_img_dir = '/mnt/data1/wx/data/train_npy/images'
train_mask_dir = '/mnt/data1/wx/data/train_npy/line_masks'
val_img_dir = '/mnt/data1/wx/data/val_npy/images'
val_mask_dir = '/mnt/data1/wx/data/val_npy/line_masks'
model_save_dir = '/mnt/data3/brain_angle_correction/model_save/no_rotate_line_masks_gray_scale1_shift1'

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
model = unet_modify.unet_modify_res_group().cuda()
# model.load_state_dict(torch.load(os.path.join(model_save_dir, 'unet_modify_group.pkl')))
# model.load_state_dict(torch.load('/mnt/data1/wx/graduation_project/model/unet_modify_group.pkl'))
loss_fn = dice_bce_loss.my_loss().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
# scheduler = MultiStepLR(optimizer=optimizer, milestones=[10,20], gamma=0.2)

min_val_loss = 100
min_dice_coef = 100
total_val_loss = []
total_train_loss = []
for epoch in range(400):
    train_loader = data.DataLoader(dataset_3d.my_dataset_seg(train_img_paths, train_mask_paths, 24, is_crop=False, is_rotate=False, is_gray_scale_shift=True), num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
    train_loss_sum = 0
    train_dice_coef_sum = 0
    # scheduler.step()
    for i,(x,y) in enumerate(train_loader):
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
        if i%2 == 0:
            print('batch num:{}, loss: {}, dice_coef: {}, min_true: {}, max_true: {}, min_pred: {}, max_pred: {}'.format(i, loss.item(), single_dice_coef, torch.min(y_true), torch.max(y_true), torch.min(y_pred), torch.max(y_pred)))
    print('------- epoch: {}, epoch loss: {}, epoch dice: {} --------'.format(epoch, train_loss_sum / len(train_loader), train_dice_coef_sum / len(train_loader)))
    total_train_loss.append(train_loss_sum / len(train_loader))
    with torch.no_grad():
        val_loader = data.DataLoader(dataset_3d.my_dataset_seg(val_img_paths, val_mask_paths, 24, is_crop=False, is_rotate=False), num_workers=4, batch_size=batch_size, shuffle=True)
        val_loss_sum = 0
        val_dice_coef_sum = 0
        for i, (x, y) in enumerate(val_loader):
            x = Variable(x).cuda()
            y_true = Variable(y).cuda()

            y_pred = model(x)
            val_loss_sum += loss_fn(y_pred, y_true).item()
            val_dice_coef_sum += dice_bce_loss.dice_coef(y_true, y_pred).item()
        print('-------- val: epoch: {}, epoch loss: {}, epoch dice_coef: {} ------'.format(epoch, val_loss_sum / len(val_loader), val_dice_coef_sum / len(val_loader)))
        total_val_loss.append(val_loss_sum / len(val_loader))
        if (val_loss_sum / len(val_loader)) < min_val_loss:
            min_dice_coef = val_dice_coef_sum / len(val_loader)
            print('----impove loss from {} to {}, dice_coef: {}--------'.format(min_val_loss, (val_loss_sum / len(val_loader)), min_dice_coef))
            min_val_loss = val_loss_sum / len(val_loader)
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'unet_modify_group.pkl'))
        else:
            print('----loss not impove from {}, dice_coef: {}'.format(min_val_loss, min_dice_coef))
    plt.figure('loss')
    plt.plot(range(0, epoch + 1), total_val_loss, label='val')
    plt.plot(range(0, epoch + 1), total_train_loss, label='train')
    plt.savefig(os.path.join(model_save_dir, 'unet_modify_group.jpg'))
    plt.close('loss')
print('----- min_val_loss: {}'.format(min_val_loss))