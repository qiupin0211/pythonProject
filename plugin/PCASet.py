import numpy as np


class PCADataset:
    """
        降维：dimension reduction 减少特征
        很少用
        打打马赛克
        思想很重要
    """

    # n_components 留下的特征数
    # 开除25人的肉身，但是，把他们的能力留下，把n个特征的信息抽取出来，放到5个身上
    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X):
        # Numpy(np.linalg.svd)进行奇异值分解
        U, S, V = np.linalg.svd(X)
        self.Vt = V[:self.n_components, :].T

    def transform(self, X):
        # 需转换的矩阵  乘  V的右奇异值指定行的转置
        return X @ self.Vt
