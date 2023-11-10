import numpy as np
from sklearn.datasets import make_regression
from sklearn.datasets import make_blobs
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import PolynomialFeatures
import os


def data_predict(X_train, X_test):
    """
        在机器学习中，有一些算法需要进行数据预处理，以提高模型的性能和准确性。以下是一些常见的算法，通常需要进行数据预处理：
        支持向量机（Support Vector Machines，SVM）：
            SVM 是一种用于分类和回归的监督学习算法。
            在使用 SVM 之前，通常需要对数据进行特征缩放，以确保各个特征具有相似的尺度。
            这是因为 SVM 是基于距离度量的算法，如果特征具有不同的尺度，可能会导致模型偏向某些特征。
        K近邻算法（K-Nearest Neighbors，KNN）：
            KNN 是一种基于实例的学习算法，用于分类和回归。
            在使用 KNN 之前，通常需要对数据进行特征缩放，以确保各个特征具有相似的尺度。
            此外，还可以对数据进行降维或特征选择，以减少计算复杂性和降低噪声影响。
        神经网络（Neural Networks）：
            神经网络是一种强大的机器学习模型，可以用于各种任务。
            在使用神经网络之前，通常需要对数据进行标准化或归一化处理，以确保各个特征具有相似的尺度。
            此外，还可以进行特征选择、降维或数据平衡等预处理步骤。
        决策树和随机森林（Decision Trees and Random Forests）：
            决策树和随机森林是一种用于分类和回归的监督学习算法。
            在使用这些算法之前，通常需要对数据进行特征缩放、编码分类变量、处理缺失值等预处理步骤。
        聚类算法（Clustering）：
            聚类算法用于将数据集中的样本分成不同的组或簇。
            在使用聚类算法之前，通常需要对数据进行特征缩放、处理缺失值、处理离群值等预处理步骤。
        需要注意的是，不是所有的机器学习算法都需要进行预处理。
        某些算法如朴素贝叶斯（Naive Bayes）和决策森林（Decision Forests）等对数据的尺度和分布不敏感，
        因此不需要进行特定的预处理步骤。预处理的具体步骤和方法取决于数据的特点和算法的要求。

        数据的预处理：
        特征缩放是数据预处理中的一个重要步骤，其目的是将不同特征的值范围调整到相似的尺度。特征缩放的主要原因包括：

        模型收敛速度更快：某些机器学习算法，如梯度下降法，对于特征值范围较大的特征更敏感。
        如果特征之间的尺度差异很大，模型可能需要更长的时间才能收敛到最优解。通过特征缩放，可以加快模型的收敛速度。

        避免特征权重偏倚：在一些机器学习算法中，特征的权重对模型的预测结果有直接影响。
        当特征之间的尺度差异很大时，具有较大尺度的特征可能会在模型中占据更大的权重，而较小尺度的特征则可能被忽略。
        通过特征缩放，可以避免特征权重的偏倚，使得模型更公平地对待各个特征。

        防止数值计算问题：在一些机器学习算法中，涉及到距离计算或优化问题的求解过程。
        如果特征之间的尺度差异很大，可能会导致数值计算上的不稳定性或数值溢出的问题。通过特征缩放，可以减少这些数值计算问题的发生。

        常见的特征缩放方法包括标准化（将数据转换为均值为0，方差为1的分布）和归一化（将数据缩放到0到1的范围）。
        选择合适的特征缩放方法取决于数据的分布和算法的要求。特征缩放有助于提高模型的性能和稳定性，使得特征之间的比较更加公平和可靠。
    """

    _mean = X_train.mean(axis=0)
    _std = X_train.std(axis=0)

    # 减去均值的操作可以使数据的均值变为0，这有助于消除不同特征之间的偏差。
    # 通过除以标准差，可以将数据的尺度缩放到相对一致的范围内，使得不同特征的方差相对平衡。
    # 这样做的好处是可以避免某些特征的数值范围较大导致模型对其更敏感，从而更好地适应各个特征之间的差异，
    # 提高模型的鲁棒性和泛化能力
    # 鲁棒性
    # 指的是模型对于输入数据中的噪声、异常值或干扰的抵抗能力。
    # 一个鲁棒性较高的模型能够在面对不完美或有干扰的数据时仍然保持良好的性能。
    # 鲁棒性的提高可以使模型对于数据中的一些变化或扰动具有更好的适应能力，
    # 从而减少模型的过拟合和对异常数据的过度敏感。

    # 泛化能力
    # 指的是模型在未见过的新数据上的表现能力。
    # 一个具有良好泛化能力的模型能够从训练数据中学习到普遍的规律和模式，
    # 并能够将这些学习到的知识应用到未见过的数据上。泛化能力的提高可以使模型在实际应用中更具有实用性，
    # 能够处理各种不同的数据样本，并具有较好的预测能力。
    X_train = (X_train - _mean) / _std
    X_test = (X_test - _mean) / _std

    results = []
    results.append(X_train)
    results.append(X_test)

    return tuple(results)


def gaussian_p(mean=None, std=None, x=None):
    """
        高斯分布的概率密度函数（Probability Density Function，PDF）
        f(x) = (1 / (σ * √(2π))) * e^(-(x-μ)^2 / (2σ^2))
        μ是均值，σ是标准差，e是自然对数的底数
    :param u:
    :param d:
    :param x:
    :return:
    """
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(- (x - mean) ** 2 / 2 / std / std)


def get_file_path(dir, file_name=None):
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = str(script_dir).replace('plugin', '')

    # 待读取文件相对于当前脚本的路径
    file_path = dir
    if file_name:
        file_path = dir + '\\' + file_name

    # 构建待读取文件的绝对路径
    abs_file_path = os.path.join(script_dir, file_path)

    return abs_file_path


def my_train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1):
    """
    :param X:
    :param y:
    :param test_size:
    :param shuffle:
    :param random_state:
    :return: X_train  X_test  y_train  y_test
    """

    # 数据集大小
    _num = len(X)

    # 判断切分的数量
    if test_size is None:  # 不要等于0
        test_size = int(0.2 * _num)

        # 实例判断
    elif isinstance(test_size, int):
        if test_size <= 0 or test_size >= _num:
            raise ValueError("test_size 有误")
    elif isinstance(test_size, float):
        if test_size <= 0.0 or test_size >= 1.0:
            raise ValueError("test_size 有误")
        else:
            test_size = int(test_size * _num)
    else:
        raise ValueError("test_size 有误")
    if random_state is not None:
        np.random.seed(random_state)

    # 打乱顺序
    indices = np.arange(_num)  # 生成一个序列 0 ~ _num
    np.random.shuffle(indices)  # 打散

    # 重新按 indices 进行装载X，y
    X_y_array = []
    X_y_array.append(X[indices])
    X_y_array.append(y[indices])

    # 切分输出
    results = []
    for temp in X_y_array:
        temp_train = temp[test_size:]
        temp_test = temp[:test_size]
        results.append(temp_train)
        results.append(temp_test)

    # 返回
    return tuple(results)


def read_bost_data():
    '''
    读取 波士顿房价预测 数据集
    '''

    X = []
    y = []
    with open(file=get_file_path("file", "boston_house_prices.csv"), mode="r", encoding="utf8") as f:
        f.readline()
        f.readline()
        for line in f:
            line = [float(ele) for ele in line.strip().split(",")]  # list列表推导式
            features = line[:-1]
            label = line[-1]
            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y


def read_ywh_data():
    '''
    读取 鸢尾花 数据集
        特征：萼片长度、萼片宽度、花瓣长度、花瓣宽度
        标签：山鸢尾（Iris setosa）、变色鸢尾（Iris versicolor）、维吉尼亚鸢尾（Iris virginica）
    '''

    X = []
    y = []
    with open(file=get_file_path("file", "iris.csv"), mode="r", encoding="utf8") as f:
        f.readline()
        for line in f:
            line = line.strip().split(",")

            # line[:-1] 表示从字符串 line 的开头到倒数第二个字符（不包括最后一个字符）的子串
            features = [float(ele) for ele in line[:-1]]
            label = int(line[-1])

            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y


def make_sklearn_regression_data():
    '''
        生成 sklearn 回归 数据集
        make_regression 用于生成回归任务的合成数据集
    '''

    # make_regression(n_samples=10000, n_features=20, n_informative=15)
    # 表示生成一个包含 10000 个样本、20 个特征的数据集，其中有 15 个特征是具有信息的，其余 5 个特征是随机生成的非信息特征
    X, y = make_regression(n_samples=10000, n_features=20, n_informative=15)

    X = np.array(X)
    y = np.array(y)

    return X, y


def make_sklearn_blobs_data():
    '''
        生成 sklearn 聚类 数据集
        make_blobs 用于生成聚类任务的合成数据集
    '''

    # make_blobs(n_samples=1000, n_features=2, centers=4, random_state=0)
    # 表示生成一个包含 1000 个样本、2 个特征的数据集，其中有 4 个簇（类别）。
    # 每个簇的数据点服从高斯分布，簇之间有明显的区分。
    # 通过调整 centers 参数的值，可以控制生成数据集中簇的数量和位置，从而调整生成数据集的复杂度和分布情况
    X, y = make_blobs(n_samples=1000, n_features=2, centers=4, random_state=0)

    X = np.array(X)
    y = np.array(y)

    return X, y


def load_breast_cancer_data():
    '''
        加载 sklearn 乳腺癌 数据集
        数据集规模：乳腺癌数据集包含569个样本（肿瘤），每个样本具有30个特征变量
        label: 0 表示良性肿瘤（无害） 1 表示恶性肿瘤（有害）
    '''

    data = load_breast_cancer()

    X = data["data"]
    y = data["target"]

    return X, y


def convert_to_2(X_train, X_test):
    """
        由 1 项式 变为 2 项式
        :param X_train:
        :param X_test:
        :return:
    """
    # 构建模型 degree=2 二项式
    cvt = PolynomialFeatures(degree=2,
                             interaction_only=False,
                             include_bias=True)
    cvt.fit(X=X_train)

    temp_train = cvt.transform(X=X_train)
    temp_test = cvt.transform(X=X_test)

    results = []
    results.append(temp_train)
    results.append(temp_test)

    # 返回
    return tuple(results)


################### TEST ####################
# X, y = read_bost_data()

# X, y = read_ywh_data()

# X, y = make_sklearn_regression_data()

# m, n = X.shape

# print("X.shape:\n", X.shape)
# print("m=:\n", m)
# print("-" * 100)
# print("X=:\n", X)
# print("-" * 100)
# print("y=:\n", y)
load_breast_cancer_data()
