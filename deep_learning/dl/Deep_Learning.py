"""
    深度学习：批量化处理
"""
import numpy as np
from sklearn.model_selection import train_test_split
from plugin import Tools as T
from plugin.Packageset import PackageDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch
import cv2


def linearR(X, y):
    '''
        - 假定：y = F(X) = w1x1 + w2x2 + ... + w13x13 + b
        - 求解 w 和 b

        - 1，随机初始化一下这些参数：w1, w2, ... w13, b
        - 2，先尝试预测一下：y_pred = F(X_train)，此时，大概率预测的是一堆垃圾 ...
        - 3，衡量这个偏差：loss = loss_fn(y_pred, y_train)
        - 4，日拱一卒，逐步缩小差距，一直到差距很小！！！
            - loss 跟 w, b 有关系， 是由 w 和 b 决定的！！！
            - loss = loss_fn(w, b)
            - 求 loss 的最小值问题
            - 问：当 w 和 b 取什么值的时候，loss 取得最小值
    '''

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    # 数据预处理：标准化，否则会因数据绝对值大而出现 nan
    X_train, X_test = T.data_predict(X_train, X_test)

    # 批量化打包数据
    # X：[batch_size, n_features]
    # y: [batch_size,1]
    train_dataset = PackageDataset(X=X_train, y=y_train)
    test_dataset = PackageDataset(X=X_test, y=y_test)

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=24)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=24)

    # 构建模型
    model = nn.Linear(in_features=np.shape(X)[1], out_features=1)

    # 准备训练
    epochs = 100  # 数据集轮回数
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2)  # 梯度优化器，控制步长
    loss_fn = nn.MSELoss()  # MSE

    # 训练
    # https://blog.csdn.net/PanYHHH/article/details/107361827
    for epoch in range(epochs):
        for X, y in train_dataloader:
            y_pred = model(X)
            loss = loss_fn(y_pred, y)  # MSE偏差

            loss.backward()  # 求偏导，因为它，所以上天  计算梯度
            optimizer.step()  # 减偏导 x0 -= (1e-2) * dfn(x0)

            optimizer.zero_grad()  # 清空梯度

    # ??
    # y_pred2 = models(torch.tensor(data=X_test, dtype=torch.float32)).sigmoid_()

    results = []

    results.append(model.weight.data)
    results.append(model.bias.data)

    # 返回
    return tuple(results)


def classify(X, y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    # 数据预处理：标准化，否则会因数据绝对值大而出现 nan
    X_train, X_test = T.data_predict(X_train, X_test)

    X_train = torch.from_numpy(X_train).to(dtype=torch.float32)
    y_train = torch.from_numpy(y_train).to(dtype=torch.float32).view(-1, 1)
    # y_train = torch.from_numpy(y_train).to(dtype=torch.long)

    # 构建模型
    model = nn.Linear(in_features=np.shape(X)[1], out_features=1)
    # models = nn.Linear(in_features=np.shape(X)[1], out_features=2)

    # 准备训练
    epochs = 1000  # 数据集轮回数

    loss_fn = nn.BCELoss()
    # loss_fn = nn.CrossEntropyLoss() # 交叉熵

    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2)  # 随机梯度下降

    # 训练
    for epoch in range(epochs):
        y_pred = model(X_train)

        loss = loss_fn(torch.sigmoid(y_pred), y_train)
        # loss = loss_fn(y_pred, y_train)  # MSE偏差

        loss.backward()  # 求偏导，因为它，所以上天  计算梯度
        optimizer.step()  # 减偏导 x0 -= (1e-2) * dfn(x0)

        optimizer.zero_grad()  # 清空梯度

    # 保存模型
    # torch.save(obj=models, f='models/d_c.lxh') # 保存数据
    torch.save(obj=model.state_dict(), f='models/d_c.pt') # 保存参数

    # 使用模型
    # model1 = nn.Linear(in_features=30, out_features=2)
    # model1.load_state_dict(state_dict=torch.load(f="models/d_c.pt"))

    X_test = torch.from_numpy(X_test).to(dtype=torch.float32)
    y_pred_1 = model(X_test)

    results = []

    results.append(model.weight.data)
    results.append(model.bias.data)

    # 返回
    return tuple(results)

def video_capture():
    """
        流式读取
        三步走策略
    """

    # 建立连接
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        status, frame = cap.read()
        if status:
            """
                添加你的算法处理代码
            """

            # 显示图像
            cv2.imshow(winname="demo", mat=frame)

            # 等待ESC按下
            if cv2.waitKey(delay=100) == 27:
                break
        else:
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()