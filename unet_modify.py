from torch import nn
from torch.nn import functional as F
import torch


# 带分组卷积的残差块
class res_block_group(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, groups, act_func=nn.ReLU(inplace=True)):
        super(res_block_group, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1, groups=groups)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv_res = nn.Conv2d(in_channels, out_channels, 1, groups=groups)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)

        res = self.conv_res(x)
        res = self.bn2(res)
        out = torch.add(out, res)
        out = self.act_func(out)

        return out


# 不带分组卷积的残差块
class res_block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(res_block, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv_res = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)

        res = self.conv_res(x)
        res = self.bn2(res)
        out = torch.add(out, res)
        out = self.act_func(out)

        return out


# 训练的时候只用到这个模型
class unet_modify_res_group(nn.Module):
    # 将模型中所有参数全都减半,24 24 48......
    def __init__(self, input_channels=12, output_channels=12, nb_filter=(24, 24, 48, 48, 96)):
        super().__init__()
        nb_groups = [12, 6, 3, 1, 1]
        self.pool00 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool01 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool12 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool23 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool34 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.up10 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], 3, 2, padding=1, output_padding=1)
        self.up21 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], 3, 2, padding=1, output_padding=1)
        self.up32 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], 3, 2, padding=1, output_padding=1)
        self.up43 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], 3, 2, padding=1, output_padding=1)

        self.conv0_0 = res_block_group(input_channels, nb_filter[0], nb_filter[0], groups=nb_groups[0])
        self.conv1_0 = res_block_group(nb_filter[0], nb_filter[1], nb_filter[1], groups=nb_groups[1])
        self.conv2_0 = res_block_group(nb_filter[1], nb_filter[2], nb_filter[2], groups=nb_groups[2])
        self.conv3_0 = res_block_group(nb_filter[2], nb_filter[3], nb_filter[3], groups=nb_groups[3])
        self.conv4_0 = res_block_group(nb_filter[3], nb_filter[4], nb_filter[4], groups=nb_groups[4])

        self.conv3_1 = res_block_group(nb_filter[3], nb_filter[3], nb_filter[3], groups=nb_groups[3])
        self.conv2_2 = res_block_group(nb_filter[2], nb_filter[2], nb_filter[2], groups=nb_groups[2])
        self.conv1_3 = res_block_group(nb_filter[1], nb_filter[1], nb_filter[1], groups=nb_groups[1])
        self.conv0_4 = res_block_group(nb_filter[0], nb_filter[0], nb_filter[0], groups=nb_groups[0])

        self.final = nn.ConvTranspose2d(nb_filter[0], output_channels, kernel_size=3, stride=2, padding=1,
                                        output_padding=1)

    def forward(self, input):
        x0_0 = self.conv0_0(self.pool00(input))
        x1_0 = self.conv1_0(self.pool01(x0_0))
        x2_0 = self.conv2_0(self.pool12(x1_0))
        x3_0 = self.conv3_0(self.pool23(x2_0))
        x4_0 = self.conv4_0(self.pool34(x3_0))

        x3_1 = self.conv3_1(torch.add(x3_0, self.up43(x4_0)))
        x2_2 = self.conv2_2(torch.add(x2_0, self.up32(x3_1)))
        x1_3 = self.conv1_3(torch.add(x1_0, self.up21(x2_2)))
        x0_4 = self.conv0_4(torch.add(x0_0, self.up10(x1_3)))

        output = self.final(x0_4)
        output = nn.Sigmoid()(output)
        return output
