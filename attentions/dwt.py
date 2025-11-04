import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import numpy as np

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2      # 选择偶数列
    x02 = x[:, :, 1::2, :] / 2      # 选择奇数列，/ 2：对切片后的张量进行除以2的操作，以进行归一化处理
    x1 = x01[:, :, :, 0::2]         # 选择偶数行
    x2 = x02[:, :, :, 0::2]         # 选择偶数行
    x3 = x01[:, :, :, 1::2]         # 选择奇数行
    x4 = x02[:, :, :, 1::2]         # 选择奇数行
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


dwt_module = DWT()
iwt_module = IWT()

x = Image.open('./1.bmp')
# x=Image.open('./mountain.png')
x = transforms.ToTensor()(x)
x = torch.unsqueeze(x, 0)
x = transforms.Resize(size=(256, 256))(x)
subbands = dwt_module(x)
LL = subbands[:, 0:3, :, :]
HL = subbands[:, 3:6, :, :]
LH = subbands[:, 6:9, :, :]
HH = subbands[:, 9:, :, :]
# 分解
title = ['LL', 'HL', 'LH', 'HH']

# 定义具有 1x1 卷积和 softmax 操作的函数
def conv1x1_softmax(LL):
    # 获取输入张量的通道数
    num_channels = LL.size(1)

    # 创建 1x1 卷积层
    conv = nn.Conv2d(num_channels, num_channels, kernel_size=1)

    # 将 LL 输入到卷积层中
    conv_LL = conv(LL)

    # 进行 softmax 操作
    softmax_LL = F.softmax(conv_LL, dim=1)
    concatenated = torch.cat([softmax_LL, LL], dim=1)
    return concatenated
# LL_conv1 = conv1x1_softmax(LL)


plt.figure()
for i in range(4):
    plt.subplot(2, 2, i + 1)
    # temp = subbands[:, 3 * i:3 * (i + 1), :, :].permute(1, 2, 0)
    temp = subbands[:, 3 * i:3 * (i + 1), :, :]
    #temp = torch.permute(subbands[0, 3 * i:3 * (i + 1), :, :], dims=[1, 2, 0])
    temp = conv1x1_softmax(temp)
    temp = temp.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    # 合并通道
    merged_image = np.sum(temp, axis=-1)

    # 标准化图像数据到范围 [0, 1]
    merged_image = merged_image / merged_image.max()
    plt.imshow(merged_image)
    plt.title(title[i])
    plt.axis('off')
plt.show()


iwt_output = iwt_module(subbands)

# Define the titles for the subplots


# Display the IWT outputs
plt.figure()

temp1 = iwt_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
plt.imshow(temp1)
plt.show()


# dwt_module = DWT()
# x = Image.open('./1.bmp')
# # x=Image.open('./mountain.png')
# x = transforms.ToTensor()(x)
# x = torch.unsqueeze(x, 0)
# x = transforms.Resize(size=(256, 256))(x)
# subbands = dwt_module(x)
#
# # 重构
# title = ['Original Image', 'Reconstruction Image']
# reconstruction_img = IWT()(subbands).cpu()
# #ssim_value = ssim(x, reconstruction_img)  # 计算原图与重构图之间的结构相似度
# #print("SSIM Value:", ssim_value)  # tensor(1.)
# show_list = [x[0].permute(1, 2, 0), reconstruction_img[0].permute(1, 2, 0)]
# #show_list = [torch.permute(x[0], dims=[1, 2, 0]), torch.permute(reconstruction_img[0], dims=[1, 2, 0])]
#
# plt.figure()
# for i in range(2):
#     plt.subplot(1, 2, i + 1)
#     plt.imshow(show_list[i])
#     plt.title(title[i])
#     plt.axis('off')
# plt.show()