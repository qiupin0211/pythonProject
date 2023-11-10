"""
    机器学习：数据是一次性处理完的
    数据比较少，没有批量的玩法
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

from plugin.PCASet import PCADataset
from plugin import Tools as T


def linearR(*args, X, y, judge_way=None, x_to_2=False):
    '''
        线性回归 sklearn LinearRegression
        线性回归是一种简单而强大的算法，但它假设输入变量和输出变量之间存在线性关系，且误差项服从正态分布。
        在实际应用中，如果数据不满足这些假设，线性回归可能不适用
        Y = β0 + β1X1 + β2X2 + ... + βnXn，其中Y是输出变量，X1到Xn是输入变量，β0到βn是模型的参数。
        线性回归的目标是找到最佳的参数值
    '''

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    # 数据预处理：标准化，否则会因数据绝对值大而出现 nan
    X_train, X_test = T.data_predict(X_train, X_test)

    # print("*args[0]=",args[0])
    if x_to_2:
        # 初始化特征数据  由 1 项 变为 2 项
        X_train, X_test = T.convert_to_2(X_train=X_train, X_test=X_test)

    # 线性回归 sklearn LinearRegression
    # 构建模型
    lr = LinearRegression()

    # 训练模型
    # lr.coef_ 权重
    # lr.intercept_ 偏置
    lr.fit(X=X_train, y=y_train)

    # 预测
    y_pred = lr.predict(X=X_test)

    if (judge_way == "ACC"):
        # 逻辑回归  处理分类问题 准确率（accuracy）
        # y_pred 线性回归 预测的结果为 ”连续量“，需手动进行划分，通过预测的概率进行分类
        y_pred_temp = (y_pred >= 0.5).astype(int)  # 线性回归+条件处理
        result = (y_pred_temp == y_test).mean()
    else:
        # MSE 评估
        result = ((y_pred - y_test) ** 2).mean()

    return result


def knc(X, y):
    '''
        存活原因：易理解
        KNeighborsClassifier 是 scikit-learn（一个常用的机器学习库）中的一个分类器类。
        它实现了 k-最近邻算法（k-nearest neighbors algorithm）用于分类任务
        k-最近邻算法是一种基于实例的学习方法，用于分类和回归问题
    '''

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

    # 数据预处理：标准化，否则会因数据绝对值大而出现 nan
    X_train, X_test = T.data_predict(X_train, X_test)

    # 分类器 sklearn KNeighborsClassifier
    # 构建模型
    knc = KNeighborsClassifier(n_neighbors=5)

    # 训练模型
    knc.fit(X=X_train, y=y_train)

    # 预测
    y_pred = knc.predict(X=X_test)

    # 计算准确率 准确率（accuracy）
    result = (y_test == y_pred).mean()

    return result


def knr(X, y):
    '''
        存活原因：易理解

        KNeighborsRegressor 是 scikit-learn 库中的一个回归算法类，用于基于最近邻的回归模型
        KNeighborsRegressor 类实现了 k 近邻回归算法，其中 k 表示要考虑的最近邻居的数量
        通过调整 n_neighbors 参数的值，我们可以控制最近邻居的数量，从而影响模型的预测性能

        过拟合（Overfitting）和欠拟合（Underfitting）是机器学习中常见的问题，涉及到模型对训练数据和测试数据的拟合程度。
            过拟合指的是模型在训练数据上表现良好，但在未见过的测试数据上表现较差的情况。过拟合通常发生在模型过于复杂或参数过多的情况下。当模型过于复杂时，它可能会过度记忆训练数据中的噪声和细节，导致在新数据上的泛化能力下降。过拟合的模型可能会出现过高的方差，即对训练数据的扰动敏感。
            欠拟合指的是模型无法很好地拟合训练数据，表现不足以捕捉数据中的关键特征和模式。欠拟合通常发生在模型过于简单或参数过少的情况下。当模型过于简单时，它可能无法捕捉到数据的复杂性和变化，导致在训练数据和测试数据上都表现不佳。欠拟合的模型可能会出现过高的偏差，即对数据的整体趋势无法准确建模。
            为了解决过拟合和欠拟合问题，可以采取以下方法：
            过拟合的解决方法：
                增加训练数据：提供更多的数据样本，可以减少模型对训练数据中噪声和细节的过度拟合。
                减少模型复杂度：简化模型结构，减少参数数量，可以降低模型的复杂性，减少过拟合的风险。
                使用正则化技术：例如 L1 正则化（Lasso）和 L2 正则化（Ridge），通过对模型的参数施加惩罚，可以控制参数的大小，防止过拟合。
                使用特征选择：选择最相关的特征，去除不相关或冗余的特征，可以减少模型的复杂性，提高泛化能力。
            欠拟合的解决方法：
                增加模型复杂度：增加模型的容量，例如增加层数、增加参数数量等，使模型能够更好地拟合数据的复杂性。
                增加特征数量：添加更多的特征，提供更多的信息和变化，以便模型能够更好地捕捉数据的模式。
                减少正则化强度：如果使用了正则化技术，可以降低正则化的强度，允许模型更好地适应训练数据。
                调整超参数：例如学习率、迭代次数等超参数，通过调整这些参数来改善模型的性能。
    '''

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

    # 数据预处理：标准化，否则会因数据绝对值大而出现 nan
    X_train, X_test = T.data_predict(X_train, X_test)

    # 分类器 sklearn KNeighborsClassifier
    # 构建模型
    knc = KNeighborsRegressor(n_neighbors=7)

    # 训练模型
    knc.fit(X=X_train, y=y_train)

    # 预测
    y_pred = knc.predict(X=X_test)

    # MSE 均方误差
    result = mean_squared_error(y_test, y_pred)

    return result


def logisticR(*args, X, y):
    '''
        逻辑回归 sklearn LogisticRegression
        实际是个二分类
        结合sigmoid函数，线性回归函数，把线性回归模型的输出作为sigmoid函数的输入。于是最后就变成了逻辑回归模型
    '''

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    # 数据预处理：标准化，否则会因数据绝对值大而出现 nan
    X_train, X_test = T.data_predict(X_train, X_test)

    # 构建模型
    logiR = LogisticRegression(max_iter=10000)

    # 训练模型
    logiR.fit(X=X_train, y=y_train)

    # 预测
    # - 假设函数：y = sigmoid(w1x1 + w2x2 + w3x3 + ... + wnxn + b)
    # - sigmoid：1 / (1 + np.exp(-y)) 输出范围：（0, 1）
    # y_pred 经过 sigmoid 函数处理
    #_y_pred = logiR.predict_proba(X=X_test)  # 返回的是概率值，未经过 sigmoid 处理，value1：是0的概率；value2：是1的概率
    y_pred = logiR.predict(X=X_test)

    # 准确率
    result = (y_pred == y_test).mean()

    return result


def gaussian(*args, X, y):
    '''
        高斯朴素贝叶斯
            - 朴素：假定所有特征都互相独立
		    - 高斯：假定所有特征都是高斯分布的
        f(x) = (1 / (σ * √(2π))) * e^(-(x-μ)^2 / (2σ^2))
        μ是均值，σ是标准差，e是自然对数的底数
        解决文本分类问题：快
    '''

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    # 数据预处理：标准化，否则会因数据绝对值大而出现 nan
    X_train, X_test = T.data_predict(X_train, X_test)

    # 构建模型
    gaussian = GaussianNB()

    # 训练模型
    gaussian.fit(X=X_train, y=y_train)

    # 预测
    y_pred = gaussian.predict(X=X_test)

    # 准确率
    result = (y_pred == y_test).mean()

    return result


def decision_tree_c(*args, X, y, pca=None):
    """
        決策樹分類
        采用了 香农 信息熵的思想，公式没有用
    :param args:
    :param X:
    :param y:
    :return:
    """

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    # 数据预处理：标准化，否则会因数据绝对值大而出现 nan
    X_train, X_test = T.data_predict(X_train, X_test)

    # 奇异值分解（Singular Value Decomposition，SVD）是一种矩阵分解技术，
    # 将一个矩阵分解为三个矩阵的乘积。SVD的三个矩阵分别是U、Σ和V^T。
    if pca == 'PCA':
        pca = PCA(n_components=5)
        pca.fit(X=X_train)

        X_train = pca.transform(X=X_train)
        X_test = pca.transform(X=X_test)
    elif pca == 'myPCA':
        myPCA = PCADataset(n_components=5)
        myPCA.fit(X=X_train)

        X_train = myPCA.transform(X=X_train)
        X_test = myPCA.transform(X=X_test)

    # 构建模型
    # criterion='gini' -log(p)*p : (1-p)*p
    dtc = DecisionTreeClassifier(criterion='gini', max_depth=4)

    # 训练模型
    dtc.fit(X=X_train, y=y_train)

    # 预测
    y_pred = dtc.predict(X=X_test)

    # 准确率
    result = (y_pred == y_test).mean()

    return result


def decision_tree_r(*args, X, y):
    """
        決策树回归
        均值：μ = (x₁ + x₂ + ... + xₙ) / n
        方差：σ² = Σ(xᵢ - μ)² / n
    :param args:
    :param X:
    :param y:
    :return:
    """

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    # 数据预处理：标准化，否则会因数据绝对值大而出现 nan
    X_train, X_test = T.data_predict(X_train, X_test)

    # 构建模型
    # max_depth=3 剪枝策略
    dtr = DecisionTreeRegressor(max_depth=3)

    # 训练模型
    dtr.fit(X=X_train, y=y_train)

    # 预测
    y_pred = dtr.predict(X=X_test)

    # MSE
    result = ((y_pred - y_test) ** 2).mean()
    # result = mean_squared_error(y_pred, y_test)

    return result


def svm(*args, X, y):
    """
        支持向量机（Support Vector Machine，SVM）
        常用的监督学习算法，主要用于分类和回归任务。
        它的目标是在特征空间中找到一个最优的超平面（对于二分类问题）或超平面集合（对于多分类问题），以将不同类别的样本分隔开来。

        SVM 在处理小样本、高维数据和非线性问题上表现出色，
        并且具有较强的泛化能力。它在实践中广泛应用于文本分类、图像识别、生物信息学等领域。

        解决卧底问题（线性不可分）, 转为高维空间来看待（ex.一楼看全是垃圾，顶楼看都不是问题），使用核函数
    :param args:
    :param X:
    :param y:
    :return:
    """

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    # 数据预处理：标准化，否则会因数据绝对值大而出现 nan
    X_train, X_test = T.data_predict(X_train, X_test)

    # 构建模型
    svm = SVC()

    # 训练模型
    svm.fit(X=X_train, y=y_train)

    # 预测
    y_pred = svm.predict(X=X_test)

    # 计算准确率 准确率（accuracy）
    result = (y_test == y_pred).mean()

    return result


def km(*args, X, y):
    """
        K-Means 算法的优点包括简单易实现、计算效率高等。然而，该算法对于初始质心的选择敏感，并且可能收敛到局部最优解。
        为了克服这些问题，可以采用多次运行 K-Means 算法并选择最优结果，或使用改进的变体算法（如 K-Means++）来选择初始质心。

        在实践中，K-Means 算法常用于聚类分析、图像分割、异常检测等任务，以及作为其他机器学习算法的预处理步骤。

        聚类 是没有标签的，有标签的称 监督学习算法

        用户分层、分类

        https://blog.csdn.net/huangfei711/article/details/78480078
    :param args:
    :param X:
    :param y:
    :return:
    """

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    # 数据预处理：标准化，否则会因数据绝对值大而出现 nan
    X_train, X_test = T.data_predict(X_train, X_test)

    # 训练前
    # c=y_train 表示 颜色的类别  =  y_train 的 centers 类 个数
    plt.scatter(x=X_train[:, 0], y=X_train[:, 1], c=y_train)
    plt.show()

    # 构建模型
    # 4 个中心 km.cluster_centers_
    km = KMeans(n_clusters=4)

    # 训练模型
    km.fit(X=X_train)

    # 预测
    km.predict(X=X_test)

    plt.scatter(x=X_train[:, 0], y=X_train[:, 1], c=y_train)
    plt.scatter(x=km.cluster_centers_[:, 0], y=km.cluster_centers_[:, 1], c="red", s=100, marker="*")
    plt.show()

    return 0
