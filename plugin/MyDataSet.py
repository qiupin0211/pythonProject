from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image  # PIL (Python Imaging Library)
import torch
import os

"""
    定义预处理相关的操作
"""


def my_preprocess(data):
    # 定义 transforms.Compose() 作为局部变量
    preprocess = transforms.Compose([
        # 将输入图像的大小调整为 100x100 像素
        transforms.Resize(size=100),

        # 将图像转换为张量形式
        transforms.ToTensor(),

        # 将图像的每个通道进行标准化，使其均值为 0，标准差为 1
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 在函数内使用 transform
    return preprocess(data)


class My_Dataset(Dataset):
    '''
        继承Dataset，自定义一个数据集，实现样本按索引读取
    '''

    def __init__(self, file_part="trian"):
        self.file_part = file_part
        self._read_raw_data()

    def _read_raw_data(self):
        self.X = []
        self.y = []
        for folder in os.listdir(self.file_part):
            label = int(folder[-1])
            folder_path = os.path.join(self.file_part, folder)
            for file in os.listdir(folder_path):
                self.X.append(os.path.join(folder_path, file))
                self.y.append(label)

    def __getitem__(self, idx):
        # 处理图像 3维张量 [C, H, W]
        img = Image.open(self.X[idx])

        img = my_preprocess(img)

        # 处理标签 0 维张量
        label = torch.tensor(data=self.y[idx], dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.X)
