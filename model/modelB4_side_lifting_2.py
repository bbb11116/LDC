# Lightweight Dense CNN for Edge Detection
# It has less than 1 Million parameters
'''
LDC side_haar
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
# from attentions.conv_pdc import Conv2d
import numpy as np
from model.lifting import LiftingScheme2D, WaveletHaar2D, WaveletHaar
from attentions.ChannelAtt import ChannelAttention
from utils.AF.Fsmish import smish as Fsmish
from utils.AF.Xsmish import Smish




class Regression(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Regression, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((30,4))
        self.classifier = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(in_ch//2, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.classifier(self.global_pool(x)).squeeze(1)

def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0,)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)

        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class CoFusion(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3,
                               stride=1, padding=1) # before 64
        self.conv3= nn.Conv2d(32, out_ch, kernel_size=3,
                               stride=1, padding=1)# before 64  instead of 32
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 32) # before 64

    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = F.softmax(self.conv3(attn), dim=1)

        # 按照通道数将attn分割，并保存为图像
        # for i in range(attn.shape[1]):
        #     attn_slice = attn[0, i, :, :]
        #     attn_slice = attn_slice.cpu().detach().numpy()
        #     attn_slice = np.array(attn_slice * 255, dtype=np.uint8)
        #     cv2.imwrite(f'./attn_{i}.png', attn_slice)


        return ((x * attn).sum(1)).unsqueeze(1)


class DoubleFusion(nn.Module):
    # TED fusion before the final edge map prediction
    def __init__(self, in_ch, out_ch):
        super(DoubleFusion, self).__init__()
        self.DWconv1 = nn.Conv2d(in_ch, in_ch*8, kernel_size=3,
                               stride=1, padding=1, groups=in_ch) # before 64
        self.PSconv1 = nn.PixelShuffle(1)

        self.DWconv2 = nn.Conv2d(32, 32*1, kernel_size=3,
                               stride=1, padding=1,groups=32)# before 64  instead of 32

        self.AF= Smish()#XAF() #nn.Tanh()# XAF() #   # Smish()#


    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.PSconv1(self.DWconv1(self.AF(x))) # #TEED best res TEDv14 [8, 32, 352, 352]

        attn2 = self.PSconv1(self.DWconv2(self.AF(attn))) # #TEED best res TEDv14[8, 3, 352, 352]

        return Fsmish(((attn2 +attn).sum(1)).unsqueeze(1)) #TED best res

class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        # self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True)),
        self.add_module('norm1', nn.BatchNorm2d(out_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True)),
        self.add_module('norm2', nn.BatchNorm2d(out_features))

    def forward(self, x):
        x1, x2 = x

        new_features = super(_DenseLayer, self).forward(F.relu(x1))  # F.relu()

        return 0.5 * (new_features + x2), x2

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features

class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads=[0,0,1,3,7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)

class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, use_ac=False):
        super(SingleConvBlock, self).__init__()
        # self.use_bn = use_bs
        self.use_ac=use_ac
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        if self.use_ac:
            self.smish = Smish()

    def forward(self, x):
        x = self.conv(x)
        if self.use_ac:
            return self.smish(x)
        else:
            return x

class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.smish= Smish()#nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.smish(x)
        x = self.conv2(x)
        if self.use_act:
            x = self.smish(x)
        return x

class SideHaarBlock(nn.Module):
    def __init__(self,in_features):
        super(SideHaarBlock, self).__init__()

        self.out_channels = in_features // 2
        self.out_features = in_features * 2

        self.CAM = ChannelAttention(in_features)
        self.down = nn.Conv2d(in_channels=in_features, out_channels=self.out_channels, kernel_size=1, stride=1)
        self.haar = WaveletHaar2D()
        self.bn = nn.BatchNorm2d(self.out_features)

    def forward(self, x):
        x = self.CAM(x)
        x = self.down(x)
        (LL, LH, HL, HH) = self.haar(x)
        x = torch.cat([HH, HL, LH, LL], dim=1)
        x = self.bn(x)
        return x

class LDC_side_lifting(nn.Module):
    """ Definition of the DXtrem network. """

    def __init__(self):
        super(LDC_side_lifting, self).__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2,)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(2, 32, 64) # [128,256,100,100]
        self.dblock_4 = _DenseBlock(3, 64, 96)# 128
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.CA1 = ChannelAttention(32)
        self.down1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, stride=1)

        self.CA2 = ChannelAttention(64)
        self.down2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1)

        self.wavelet1 = LiftingScheme2D(in_planes=8, share_weights=False)
        self.wavelet2 = LiftingScheme2D(in_planes=16, share_weights=False)

        # self.wavelet1 = LiftingScheme2D(in_planes=32, share_weights=False)
        # self.wavelet2 = LiftingScheme2D(in_planes=64, share_weights=False)

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.side_2 = SingleConvBlock(32, 64, 2)

        self.side_haar1 = SideHaarBlock(16)
        self.side_haar2 = SideHaarBlock(32)
        self.side_haar3 = SideHaarBlock(64)

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(32, 64, 2)
        self.pre_dense_3 = SingleConvBlock(32, 64, 1)
        self.pre_dense_4 = SingleConvBlock(64, 96, 1)# 128

        # USNet
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(64, 2)
        self.up_block_4 = UpConvBlock(96, 3)# 128
        # self.block_cat = SingleConvBlock(4, 1, stride=1, use_bs=False) # hed fusion method
        self.block_cat = DoubleFusion(4,4)# cats fusion method


        self.apply(weight_init)

    def slice(self, tensor, slice_shape):
        t_shape = tensor.shape
        height, width = slice_shape
        if t_shape[-1]!=slice_shape[-1]:
            new_tensor = F.interpolate(
                tensor, size=(height, width), mode='bicubic',align_corners=False)
        else:
            new_tensor=tensor
        # tensor[..., :height, :width]
        return new_tensor

    def forward(self, x):
        assert x.ndim == 4, x.shape
         # supose the image size is 352x352
        # Block 1
        block_1 = self.block_1(x) # [8,16,176,176]
        # block_1_side = self.side_1(block_1) # 16 [8,32,88,88]

        block_1_side = self.side_haar1(block_1) # 16 [8,32,88,88]

        # Block 2
        block_2 = self.block_2(block_1) # 32 # [8,32,176,176]
        # block_2_down = self.maxpool(block_2) # [8,32,88,88]
        block_2 = self.CA1(block_2)
        block_2_14 = self.down1(block_2)
        (c1, d1, LL1, LH1, HL1, HH1) = self.wavelet1(block_2_14)
        block_2_down = torch.cat([HH1, HL1, LH1, LL1], dim=1)
        block_2_down = self.CA1(block_2_down)

        # block_2_down = self.wavelet1(block_2)[-1]
        block_2_add = block_2_down + block_1_side # [8,32,88,88]
        # block_2_side = self.side_2(block_2_add) # [8,64,44,44] block 3 R connection

        block_2_side = self.side_haar2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down) # [8,64,88,88] block 3 L connection
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense]) # [8,64,88,88]
        # block_3_down = self.maxpool(block_3) # [8,64,44,44]
        block_3 = self.CA2(block_3)
        block_3_14 = self.down2(block_3)
        (c2, d2, LL2, LH2, HL2, HH2) = self.wavelet2(block_3_14)
        block_3_down = torch.cat([HH2, HL2, LH2, LL2], dim=1)
        block_3_down = self.CA2(block_3_down)


        # block_3_down = self.wavelet2(block_3)[-1]
        block_3_add = block_3_down + block_2_side # [8,64,44,44]

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down) # [8,64,44,44]
        block_4_pre_dense = self.pre_dense_4(block_3_down+block_2_resize_half) # [8,96,44,44]
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense]) # [8,96,44,44]

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        # results = [out_1, out_2, out_3, out_4, out_5, out_6]
        results = [out_1, out_2, out_3, out_4]

        # 将results中的每个结果保存为图像


        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx4xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        # return results
        results.append(block_cat)
        return results


if __name__ == '__main__':
    # batch_size = 8
    # img_height = 352
    # img_width = 352
    #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # input = torch.rand(batch_size, 3, img_height, img_width).to(device)
    # # target = torch.rand(batch_size, 1, img_height, img_width).to(device)
    # print(f"input shape: {input.shape}")
    # model = LDC().to(device)
    # output = model(input)
    # print(f"output shapes: {[t.shape for t in output]}")

    # for i in range(20000):
    #     print(i)
    #     output = model(input)
    #     loss = nn.MSELoss()(output[-1], target)
    #     loss.backward()

    import torchvision.transforms.functional as TF
    from torchvision import transforms

    model = LDC_side_lifting().to(device)
    image = cv2.imread('./8_4.png')
    # 调整图像尺寸
    # image = cv2.resize(image, (256, 256))  # 调整为256x256的尺寸
    # 将图像从BGR格式转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 转换为张量
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  # 标准化张量
    img = transform(image).unsqueeze(0)  # 在第0维添加一个维度，以匹配原始代码中的形状
    img = img.cuda()

    out = model(img)
    for i, result in enumerate(out):
        result = result.squeeze(0)
        result = TF.to_pil_image(result)
        result.save(f'./result_{i}.png')
