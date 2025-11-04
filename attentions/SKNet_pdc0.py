'''
SKNet: Selective Kernel Networks
Paper: https://arxiv.org/abs/1903.06586
data：2024-06-14
author：Xie
'''

import torch
from torch import nn
from .conv_pdc0 import Conv2d

class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])

        # cv
        self.convs.append(nn.Sequential(
            Conv2d('cv', features, features, kernel_size=3, stride=stride, padding=1, groups=G),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False)
        ))
        # cd
        self.convs.append(nn.Sequential(
            Conv2d('cd', features, features, kernel_size=3, stride=stride, padding=1, groups=G),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False)
        ))
        # ad
        self.convs.append(nn.Sequential(
            Conv2d('ad', features, features, kernel_size=3, stride=stride, padding=1, groups=G),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False)
        ))
        # rd
        self.convs.append(nn.Sequential(
            Conv2d('rd', features, features, kernel_size=3, stride=stride, padding=1, groups=G),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False)
        ))

        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])

        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

if __name__ == '__main__':
    x = torch.rand(8, 8, 320, 320)
    conv = SKConv(8, 32, 4, 8, 2)
    out = conv(x)
    print('out shape : {}'.format(out.shape))
