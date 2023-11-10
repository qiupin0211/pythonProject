import numpy as np
from collections import Counter


class MyKNeighborsRegression:
    def __init__(self, n_neighbors=5):
        """
            :param n_neighbors:
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
            训练函数
            - 惰性计算
        """
        # 转类型  数组
        X = np.array(X)
        y = np.array(y)

        # 把训练集挂到模型上面
        self.X = X
        self.y = y

    def predict(self, X):
        X = np.array(X)

        results = []
        for x in X:
            # 计算两个向量的距离
            distances = ((self.X - x) ** 2).sum(axis=1)

            # 取最近的 n_neighbors
            indices = distances.argsort(axis=0)[:self.n_neighbors]
            labels = self.y[indices]

            # 获取 n_neighbors 连续量的 平均值
            y_pred = labels.mean()

            results.append(y_pred)

        return np.array(results)
