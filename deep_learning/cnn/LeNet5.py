import torch
from torch import nn

'''
    LeNet-5、VGG16 和 ResNet-50 是三种经典的卷积神经网络（Convolutional Neural Networks, CNNs）
    应用于图像分类
    LeNet-5 是由 Yann LeCun 在 1998 年提出的第一个成功应用于手写数字识别的卷积神经网络。
    卷积层 + 池化层 + 全连接层
    LeNet-5 的结构相对简单，适用于较小的图像分类任务
    
    几千到几万个样本
'''

class ConvBlock(nn.Module):
    """
        打包卷积块
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=5,
                                stride=1,
                                padding=0)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LeNet_5(nn.Module):
    """
       自定义LeNet5
    """

    def __init__(self, n_classes=10):
        """
            1, 初始化父类
            2，处理超参
            3，定义需要的层
        """
        # 先初始化父类，然后再初始化子类
        super(LeNet_5, self).__init__()

        # 特征抽取
        self.feature_extractor = nn.Sequential(
            # 3 -> 6 - 6 -> 16 - 16
            # 3 卷积 6 池化 6 卷积 16 池化 16
            ConvBlock(in_channels=3, out_channels=6),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ConvBlock(in_channels=6, out_channels=16),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # 分类输出
        self.classifier = nn.Sequential(
            # 16 * 22 * 22 -> 120 -> 84 -> 10
            # 16 * 22 * 22 全连接 120 全连接 84 全连接 10
            nn.Linear(in_features=16 * 22 * 22, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=n_classes)
        )

    def forward(self, x):
        """
            正向传播
        """
        # 抽取特征
        x = self.feature_extractor(x)
        # 转换维度
        x = x.view(x.size(0), -1)
        # 分类任务
        x = self.classifier(x)
        return x
