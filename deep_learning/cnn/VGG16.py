'''
    VGG16 是由牛津大学的研究团队于 2014 年提出的卷积神经网络模型。
    它的名称来源于该模型的深度，包含了 16 层卷积层和全连接层。
    VGG16 的特点是使用了相对较小的卷积核（3x3），多次堆叠以增加网络深度，从而提高了模型的表达能力。
    VGG16 在图像分类任务中表现出色，但模型较大，需要更多的计算资源和训练时间

    数万到数十万个样本
'''

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VGG_16(nn.Module):
    def __init__(self):
        super(VGG_16, self).__init__()

        # 特征抽取
        self.feature_extractor = nn.Sequential(
            # 第 1 阶段【2卷积1池化】
            nn.Sequential(
                ConvBlock(in_channels=3, out_channels=64),
                ConvBlock(in_channels=64, out_channels=64)
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 第 2 阶段【2卷积1池化】
            nn.Sequential(
                ConvBlock(in_channels=64, out_channels=128),
                ConvBlock(in_channels=128, out_channels=128)
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 第 3 阶段【3卷积1池化】
            nn.Sequential(
                ConvBlock(in_channels=128, out_channels=256),
                ConvBlock(in_channels=256, out_channels=256),
                ConvBlock(in_channels=256, out_channels=256)
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 第 4 阶段【3卷积1池化】
            nn.Sequential(
                ConvBlock(in_channels=256, out_channels=512),
                ConvBlock(in_channels=512, out_channels=512),
                ConvBlock(in_channels=512, out_channels=512)
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 第 5 阶段【3卷积1池化】
            nn.Sequential(
                ConvBlock(in_channels=512, out_channels=512),
                ConvBlock(in_channels=512, out_channels=512),
                ConvBlock(in_channels=512, out_channels=512)
            ),
            nn.AdaptiveAvgPool2d(output_size=7)
        )

        # 输出【3全连接】
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.out(x)
        return x
