# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from deep_learning.dl import Deep_Learning as dl
from deep_learning.RunModel import RunM
from deep_learning import RunModel


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def predict(X, model):
    """
        预测函数
    """
    X = torch.from_numpy(X).to(dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred >= 0.5).to(dtype=torch.long)
        return y_pred.view(-1).numpy()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')

    # ---------------------------------------------------
    ##################### main ######################

    ########### begin 波士顿房价预测 线性回归 ############

    # # 加载数据集
    # X, y = T.read_bost_data()
    #
    # result = myMA.linearR(X=X, y=y)  # 自定义
    # print("do myself=\n", result)
    #
    # result = ml.linearR(X=X, y=y)  # 机器学习-线性回归
    # print("机器学习-线性回归=\n", result)
    #
    # result = ml.linearR(X=X, y=y, x_to_2=True)  # 机器学习-初始化特征数据  由 1 项式 变为 2 项式
    # print("机器学习-初始化特征数据  由 1 项式 变为 2 项式=\n", result)
    #
    # result = dl.linearR(X=X, y=y)  # 深度学习-梯度下降
    # print("深度学习-梯度下降=\n", result)

    # result = ml.decision_tree_r(X=X, y=y)  # 決策樹回归
    # print("決策树回归=\n", result)

    # result = cd.linearR(X=X, y=y)  # 深度学习-自定义-线性回归
    # print("深度学习-自定义-线性回归=\n", result)

    # result = cdl.linearR_Net(X=X, y=y)  # 深度学习-自定义-3层-线性回归
    # print("深度学习-自定义-3层-线性回归=\n", result)

    ########### end 波士顿房价预测 线性回归 ############

    ########### begin 鸢尾花预测 分类 ############

    # X, y = T.read_ywh_data()

    # result = myMA.linearR(X=X, y=y)  # 自定义
    # print("do myself=\n", result)
    #
    # result = ml.knc(X=X, y=y)  # 机器学习-knc
    # print("机器学习-knc=\n", result)

    # result = ml.gaussian(X=X, y=y)  # 高斯朴素贝叶斯
    # print("高斯朴素贝叶斯=\n", result)
    #
    # result = ml.decision_tree_c(X=X, y=y)  # 決策樹分類
    # print("決策樹分類=\n", result)

    ########### end 鸢尾花预测 分类 ############

    ########### begin knr 回归 ############

    # X, y = T.make_sklearn_regression_data()
    #
    # result = myMA.knr(X=X, y=y)  # 自定义
    # print("do myself=\n", result)
    #
    # result = ml.knr(X=X, y=y)  # 机器学习-knr
    # print("机器学习-knr=\n", result)
    #
    # ########### end knr 回归 ############

    ########### begin 乳腺癌数据集 分类 ############

    # 加载数据集
    # X, y = T.load_breast_cancer_data()

    # result = ml.knc(X=X, y=y)  # 机器学习-knc
    # print("机器学习-knc=\n", result)
    #
    # result = ml.linearR(X=X, y=y, judge_way="ACC")  # 机器学习-线性回归+条件处理
    # print("机器学习-线性回归+条件处理=\n", result)

    # result = ml.logisticR(X=X, y=y)  # 机器学习-逻辑回归
    # print("机器学习-逻辑回归=\n", result)

    # result = dl.linearR(X=X, y=y)  # 深度学习-梯度下降
    # print("深度学习-梯度下降=\n", result)

    # result = ml.decision_tree_c(X=X, y=y)  # 決策树分类
    # print("決策树分类=\n", result)

    # result = ml.decision_tree_c(X=X, y=y, pca='PCA')  # 決策树分类 PCA降维
    # print("決策树分类 PCA降维=\n", result)

    # result = ml.decision_tree_c(X=X, y=y, pca='myPCA')  # 決策树分类 myPCA降维
    # print("決策树分类 myPCA降维=\n", result)

    # result = ml.svm(X=X, y=y)  # SVM支持向量机
    # print("SVM支持向量机=\n", result)

    # result = dl.classify(X=X, y=y)  # 深度学习-分类
    # print("深度学习-分类=\n", result)

    # result = cdl.class_Net(X=X, y=y)  # 深度学习-自定义-3层-分类
    # print("深度学习-自定义-3层-分类=\n", result)

    ########### end 乳腺癌数据集 分类 ############

    ########### begin 聚类 数据集 ############

    # 生成数据集
    # X, y = T.make_sklearn_blobs_data()
    #
    # result = ml.km(X=X, y=y)  # K-Means 算法
    # print("K-Means 算法=\n", result)

    ########### end 聚类 数据集 ############

    ############## begin 加载模型 及 预测 #############
    # X, y = T.load_breast_cancer_data()
    #
    # _mean = X.mean(axis=0)
    # _std = X.std(axis=0)
    #
    # X_train = (X - _mean) / _std
    #
    # models = nn.Linear(in_features=30, out_features=1)
    # models.load_state_dict(state_dict=torch.load(f="models/d_c.pt"))
    # y_pred = predict(X_train, models)
    # print(y_pred)
    # print("*" * 100)
    # print(y)
    ############## begin 加载模型 及 预测 #############

    print(torch.__version__)
    # print(T.get_file_path(file_name='boston_house_prices.csv'))

    ############### begin 测试 dl cnn ################
    ### begin LeNet5
    # rm = RunM(model_type="LeNet5", epochs=60)
    # rm._train()
    #
    # # ./gestures/test/G5/IMG_1204.JPG
    # result = RunModel.predict(model_type="LeNet5", img_file="file/hand_gestures/test/G5/IMG_1204.JPG")
    # print("dl cnn result:\n", result)
    ### end LeNet5

    ### begin VGG16
    # rm = RunM(model_type="VGG16", epochs=10)
    # rm._train()
    #
    # # ./gestures/test/G5/IMG_1204.JPG
    # result = RunModel.predict(model_type="VGG16", img_file="file/hand_gestures/test/G5/IMG_1204.JPG")
    # print("dl cnn result:\n", result)
    ### end VGG16

    ### begin ResNet50
    rm = RunM(model_type="ResNet50", epochs=10)
    rm._train()

    # ./gestures/test/G5/IMG_1204.JPG
    result = RunModel.predict(model_type="ResNet50", img_file="file/hand_gestures/test/G5/IMG_1204.JPG")
    print("dl cnn result:\n", result)
    ### end ResNet50

    ############### end 测试 dl cnn ################

    ############### begin 测试 cv ################
    # img1 = cv2.imread(filename="./file/street.jpeg")
    #
    # cv2.imshow(winname="demo", mat=img1)
    #
    # # 案件等待
    # key = cv2.waitKey(delay=10000)
    #
    # if key == 27:
    #     print("按下了ESC 。。。")
    # else:
    #     print(f"按键了，但是不是ESC,是 {chr(key)}")
    #
    # cv2.destroyAllWindows()

    ############### end 测试 cv ################

    # dl.video_capture()


