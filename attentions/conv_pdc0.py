import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, pdc_type, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc_type = pdc_type

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.pdc_type == 'cv':
            pdc = PDC(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return pdc.type_cv()
        if self.pdc_type == 'cd':
            pdc = PDC(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return pdc.type_cd()
        if self.pdc_type == 'ad':
            pdc = PDC(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return pdc.type_ad()
        if self.pdc_type == 'rd':
            pdc = PDC(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return pdc.type_rd()

class PDC():
    def __init__(self, input, weight, bias, stride=1, padding=0, dilation=1, groups = 1):
        super(PDC, self).__init__()
        self.input = input
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
    def type_cv(self):
        return F.conv2d(self.input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def type_cd(self):
        weights_c = self.weight.sum(dim=[2, 3], keepdim=True)
        yc = F.conv2d(self.input, weights_c, stride=self.stride, padding=0, groups=self.groups)
        y = F.conv2d(self.input, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return y - yc

    def type_ad(self):
        shape = self.weight.shape
        weights = self.weight.view(shape[0], shape[1], -1)
        weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise
        y = F.conv2d(self.input, weights_conv, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return y

    def type_rd(self):
        padding = 2 * self.dilation

        shape = self.weight.shape
        if self.weight.is_cuda:
            buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
        else:
            buffer = torch.zeros(shape[0], shape[1], 5 * 5)

        # 这里更改cuda,与主程序的device相匹配
        device = torch.device("cuda:0")
        buffer = buffer.to(device)

        weights = self.weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
        buffer[:, :, 12] = 0
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        y = F.conv2d(self.input, buffer, self.bias, stride=self.stride, padding=padding, dilation=self.dilation, groups=self.groups)
        return y


if __name__ == '__main__':
    batch_size = 8
    img_height = 352
    img_width = 352
    #
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    input = torch.rand(batch_size, 3, img_height, img_width).to(device)
    # target = torch.rand(batch_size, 1, img_height, img_width).to(device)
    print(f"input shape: {input.shape}")
    pdc_type = 'rd'
    conv1 = Conv2d(pdc_type,3,3,kernel_size=3,  padding=1, groups=3, bias=False)

    output = conv1(input)
    print(f"output shapes: {[t.shape for t in output]}")