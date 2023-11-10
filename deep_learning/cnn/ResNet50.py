'''
    ResNet-50 是由微软研究院的研究团队于 2015 年提出的深度残差网络模型。
    ResNet-50 的主要创新是引入了残差连接，
    允许网络在训练过程中学习残差映射，解决了深层网络训练过程中的梯度消失和梯度爆炸问题。
    ResNet-50 由 50 层卷积层和全连接层组成，相较于传统的卷积神经网络，
    它具有更深的网络结构和更强的表达能力，在图像分类和其他计算机视觉任务上取得了显著的性能提升。

    数十万到数百万个样本
'''

import torch
from torch import nn
from torch.nn import functional as F

"""
    定义一个卷积短接块
"""


class ConvBlock(nn.Module):
    def __init__(self, channels, stride):
        """
            channels：指的是第一个卷积层的输出通道数
            stride：指的是第二个卷积的stride
        """
        super(ConvBlock, self).__init__()
        # 特征提取块
        self.body = nn.Sequential(
            # 1 x 1
            nn.Conv2d(in_channels=channels if stride == 1 else channels * 2,
                      out_channels=channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),

            # 3 X 3
            nn.Conv2d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),

            # 1 x 1
            nn.Conv2d(in_channels=channels,
                      out_channels=channels * 4,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(num_features=channels * 4)
        )

        # 卷积短接块
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=channels if stride == 1 else channels * 2,
                      out_channels=channels * 4,
                      kernel_size=1,
                      stride=stride,
                      padding=0),
            nn.BatchNorm2d(num_features=channels * 4)
        )

    def forward(self, x):
        x1 = self.body(x)
        x2 = self.shortcut(x)
        out = F.relu(x1 + x2)
        return out

    """
    定义一个直连块
"""


class IdentityBlock(nn.Module):
    def __init__(self, channels):
        """
            channels：指的是第一个卷积层的输出通道数
        """
        super(IdentityBlock, self).__init__()
        # 特征提取块
        self.body = nn.Sequential(
            # 1 x 1
            nn.Conv2d(in_channels=channels * 4,
                      out_channels=channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),

            # 3 X 3
            nn.Conv2d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),

            # 1 x 1
            nn.Conv2d(in_channels=channels,
                      out_channels=channels * 4,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(num_features=channels * 4)
        )

    def forward(self, x):
        x = self.body(x) + x
        out = F.relu(x)
        return out


class ResNet_50(nn.Module):
    def __init__(self):
        super(ResNet_50, self).__init__()

        # 头部处理
        self.head = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 特征提取部分
        self.body = nn.Sequential(
            # stage1
            nn.Sequential(
                ConvBlock(channels=64, stride=1),
                IdentityBlock(channels=64),
                IdentityBlock(channels=64)
            ),
            # stage2
            nn.Sequential(
                ConvBlock(channels=128, stride=2),
                IdentityBlock(channels=128),
                IdentityBlock(channels=128),
                IdentityBlock(channels=128)
            ),

            # stage3
            nn.Sequential(
                ConvBlock(channels=256, stride=2),
                IdentityBlock(channels=256),
                IdentityBlock(channels=256),
                IdentityBlock(channels=256),
                IdentityBlock(channels=256),
                IdentityBlock(channels=256)
            ),

            # stage4
            nn.Sequential(
                ConvBlock(channels=512, stride=2),
                IdentityBlock(channels=512),
                IdentityBlock(channels=512)
            )
        )

        # 输出
        self.foot = nn.Sequential(
            # 在一定程度上，解决图像任意大小问题
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=2048, out_features=1000)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.foot(x)
        return x
