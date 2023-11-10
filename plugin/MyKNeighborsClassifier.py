import numpy as np
from collections import Counter


class MyKNeighborsClassifier:
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

        # 基本校验
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("输入的参数维度有错误!!!!")

        # 把训练集挂到模型上面
        self.X = X
        self.y = y

    def predict(self, X):
        """
            预测函数
                - 第一步：先从训练集中，找出跟待预测的样本很类似的 n 个样本（欧式距离）
                - 第二步：这 n 样本中，哪个类别出现的最多，这个样本就属于哪个类别
        """
        # 类型转换
        X = np.array(X)

        # 基本校验
        if X.ndim != 2:
            raise ValueError("输入参数必须是二维的！！！")

        '''
            样本的相似度计算

            - x:[x1, x2, ..., xn]

            - 从线性代数的视角出发，认为每个样本都是一个 n 维的向量
                - 样本的相似度就可以用向量的相似度来表达
                    - 用向量的余弦相似度来表达
                     - cosine_similarity = a @ b / |a| / |b|

            - 从欧式空间的视角出发，认为每个样本都是一个 n 维空间内的点
                - 样本的相似度就可以用点的欧式距离来表达
                    - 欧式距离越小，代表越相似
                        - d = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 +....)
        '''
        # 求距离
        results = []
        # for idx, x in enumerate(X):
        for x in X:
            # 求出待测样本跟训练集中所有样本的距离
            distances = ((self.X - x) ** 2).sum(axis=1)  # 不关心距离，所以不用开根号
            # distances = np.sqrt(((self.X - x) ** 2).sum(axis=1))

            # 求出距离最近（最相似）的 n_neighbors 个邻居，返回的是邻居标签
            # argsort 默认升序
            # distances.argsort(axis=0) 表示按照垂直轴（axis=0）对 distances 进行排序，并返回排序后的索引数组
            # distances[:5] 表示对 distances 数组进行切片操作，获取数组的前5个元素
            indices = distances.argsort(axis=0)[:self.n_neighbors]
            labels = self.y[indices]

            # 投票机制 top 1
            # Counter(labels)  :  Counter({2: 3, 1: 1, 0: 1})
            label = Counter(labels).most_common(1)[0][0]  # 取次数最多的
            results.append(label)

        return np.array(results)
