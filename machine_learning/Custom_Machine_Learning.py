from sklearn.metrics import mean_squared_error
from plugin import Tools as T
from plugin import MyLinearRegression as myLR
from plugin import MyKNeighborsClassifier as myKnc
from plugin import MyKNeighborsRegression as myKnr

'''
    线性回归 sklearn LinearRegression
    线性回归是一种简单而强大的算法，但它假设输入变量和输出变量之间存在线性关系，且误差项服从正态分布。
    在实际应用中，如果数据不满足这些假设，线性回归可能不适用
    Y = β0 + β1X1 + β2X2 + ... + βnXn，其中Y是输出变量，X1到Xn是输入变量，β0到βn是模型的参数。
    线性回归的目标是找到最佳的参数值
'''


def linearR(X, y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = T.my_train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    # 线性回归
    # 构建模型
    lr = myLR.MyLinearRegression()

    # 训练模型
    # lr.coef_ 权重
    # lr.intercept_ 偏置
    lr.fit(X=X_train, y=y_train)

    # 预测
    y_pred = lr.predict(X=X_test)

    # MSE 评估
    result = ((y_pred - y_test) ** 2).mean()

    return result


def knc(X, y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = T.my_train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

    # 线性回归
    # 构建模型
    knc = myKnc.MyKNeighborsClassifier(n_neighbors=5)

    # 训练模型
    knc.fit(X=X_train, y=y_train)

    # 预测
    y_pred = knc.predict(X=X_test)

    # 评估
    result = (y_pred == y_test).mean()

    return result


def knr(X, y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = T.my_train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

    # 线性回归
    # 构建模型
    knr = myKnr.MyKNeighborsRegression(n_neighbors=5)

    # 训练模型
    knr.fit(X=X_train, y=y_train)

    # 预测
    y_pred = knr.predict(X=X_test)

    # 评估
    result = mean_squared_error(y_test, y_pred)

    return result
