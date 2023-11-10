import numpy as np


class MyLinearRegression:
    def __init__(self):
        pass

    def fit(self, X, y):
        """
            训练
                - 挖掘规律
                - w, b
                - y = Xw + 1*b   需把X加一列b
                - Xw = y
                - w=yX^T (XX^T )^(-1)
        """

        # 先转类型
        X = np.array(X)
        y = np.array(y)

        if len(X) != len(y):
            raise ValueError("X 和 y 的批量维度不相等！")

        m, n = X.shape  # 404, 13

        # 增加一列
        X = np.concatenate((X, np.ones((m, 1))), axis=1)
        y = y.reshape((m, 1))

        # 利用最小二乘法来求解 权重和偏置  np.linalg.inv(X.T @ X)  求 矩阵的逆矩阵
        # https://www.eet-china.com/mp/a27601.html
        w = np.linalg.inv(X.T @ X) @ X.T @ y

        # 挂载到 模型 上面
        self.w = w

    def predict(self, X):
        """
            推理
        """

        # 基本校验
        if not hasattr(self, "w"):  # self 里面有没有 w 这个属性
            raise AttributeError("请先训练，然后再来预测！")

        X = np.array(X)
        if X.ndim != 2:
            raise ValueError("X必须是二维的")

        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

        y_pred = X @ self.w
        y_pred = y_pred.reshape(-1)  # 改成一维
        return y_pred
