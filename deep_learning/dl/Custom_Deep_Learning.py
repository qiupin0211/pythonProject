from plugin import Tools as T
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler


def linearR(X, y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = T.my_train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

    # 预处理
    X_train, X_test = T.data_predict(X_train, X_test)
    y_train, y_test = T.data_predict(y_train, y_test)

    # 数据转张量
    X_train = torch.from_numpy(X_train).to(dtype=torch.float32)
    X_test = torch.from_numpy(X_test).to(dtype=torch.float32)

    y_train = torch.from_numpy(y_train).to(dtype=torch.float32).view(-1, 1)
    y_test = torch.from_numpy(y_test).to(dtype=torch.float32).view(-1, 1)

    # 定义模型的参数
    w = torch.randn(X_train.shape[1], 1, dtype=torch.float32, requires_grad=True)
    b = torch.randn(1, dtype=torch.float32, requires_grad=True)

    epochs = 1000
    learning_rate = 1e-3
    for epoch in range(epochs):
        # 正向传播 模型定义的处理逻辑
        y_pred = X_train @ w + b
        # 计算损失
        loss = ((y_pred - y_train) ** 2).mean()

        # 反向传播
        loss.backward()

        # 优化一步
        w.data -= learning_rate * w.grad.data
        b.data -= learning_rate * b.grad.data

        # 清空梯度
        w.grad.data.zero_()
        b.grad.data.zero_()

    results = []

    results.append(w)
    results.append(b)

    # # 预测
    y_pred = X_test @ w + b

    # MSE 评估
    result = ((y_pred - y_test) ** 2).mean()
    results.append(result)

    # 返回
    return tuple(results)


class Three_Net(nn.Module):
    """
        定义一个 3 层的全连接网络
    """

    def __init__(self, in_num=13, out_num=1):
        super(Three_Net, self).__init__()
        self.fc1 = nn.Linear(in_features=in_num, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=6)
        self.fc3 = nn.Linear(in_features=6, out_features=out_num)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def linearR_Net(X, y):
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = T.my_train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

    # 数据转张量
    # 转换为PyTorch的张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # 定义模型
    model = Three_Net(out_num=1)
    print(model)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 1000
    for epoch in range(epochs):
        # 正向传播 模型定义的处理逻辑
        y_pred = model(X_train)

        # 计算损失
        loss = loss_fn(y_pred, y_train)

        # 反向传播 和 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    results = []

    y_pred_1 = model(X_test)

    print(y_test)
    print('-' * 100)
    print(y_pred_1)

    # results.append(models.)
    # results.append(b)

    # 返回
    return tuple(results)


def class_predict(X, model):
    """
        预测函数
    """

    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred >= 0.5).to(dtype=torch.long)
        return y_pred.view(-1).numpy()


def class_Net(X, y):
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = T.my_train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

    # 数据转张量
    # 转换为PyTorch的张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 定义模型
    model = Three_Net(in_num=X.shape[1], out_num=2)
    print(model)

    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 1000
    for epoch in range(epochs):
        # 正向传播 模型定义的处理逻辑
        y_pred = model(X_train)

        # 计算损失
        loss = loss_fn(y_pred, y_train)

        # 反向传播 和 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_pred_1 = class_predict(X_test, model)

    print(y_test)
    print('-' * 100)
    print(y_pred_1)

    results = []

    # results.append(models.)
    # results.append(b)

    # 返回
    return tuple(results)
