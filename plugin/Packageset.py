import torch


class PackageDataset:
    """
        封装数据集
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        x = torch.tensor(data=self.X[idx], dtype=torch.float32)
        y = torch.tensor(data=[self.y[idx]], dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.X)
