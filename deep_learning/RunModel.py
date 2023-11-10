import torch
from torch import nn
from plugin import MyDataSet
from torch.utils.data import DataLoader
from deep_learning.cnn.LeNet5 import LeNet_5
from deep_learning.cnn.VGG16 import VGG_16
from deep_learning.cnn.ResNet50 import ResNet_50
from PIL import Image
import time, os
from plugin import Tools as T


def loaderData(dir):
    '''
        打包数据
    :return:
    '''
    train_dataset = MyDataSet.My_Dataset(file_part=f'{dir}/train')
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=True,
                                  batch_size=64)
    test_dataset = MyDataSet.My_Dataset(file_part=f'{dir}/test')
    test_dataloader = DataLoader(dataset=test_dataset,
                                 shuffle=False,
                                 batch_size=64)

    results = []
    results.append(train_dataloader)
    results.append(test_dataloader)

    # 返回
    return tuple(results)


class RunM():

    def __init__(self, last_pt=None, model_type=None, epochs=50):
        self.train_dataloader, self.test_dataloader = loaderData(T.get_file_path("file/hand_gestures", ""))

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if model_type == "LeNet5":
            self.model_type = "LeNet_5_"
            self.model = LeNet_5(n_classes=10).to(device=self.device)
        elif model_type == "VGG16":
            self.model_type = "VGG_16_"
            self.model = VGG_16().to(device=self.device)
        elif model_type == "ResNet50":
            self.model_type = "ResNet_50_"
            self.model = ResNet_50().to(device=self.device)
        else:
            self.model_type = "LeNet_5_"
            self.model = LeNet_5(n_classes=10).to(device=self.device)

        # 加载上次的训练结果
        if last_pt:
            state_dict = torch.load(T.get_file_path("models", self.model_type + "last.pt"),
                                    map_location=torch.device('cpu'))  # 加载权重参数
            self.model.load_state_dict(state_dict)  # 将权重参数加载到模型中

        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()
        self.epochs = epochs

    """
        定义过程监控
    """

    def _get_acc(self, dataloader):
        self.model.eval()  # 模型进入评估模式
        accs = []
        with torch.no_grad():
            for X, y in dataloader:
                # 数据搬家
                X = X.to(device=self.device)
                y = y.to(device=self.device)

                y_pred = self.model(X)
                y_pred = torch.argmax(input=y_pred, dim=1)
                acc = (y_pred == y).to(torch.float32).mean().item()
                accs.append(acc)
        acc = torch.tensor(data=accs, dtype=torch.float32).mean().item()
        return round(number=acc, ndigits=6)

    """
        定义训练过程
    """

    def _train(self):
        ckpt = T.get_file_path("models", "")
        cur_acc = 0
        for epoch in range(self.epochs):
            self.model.train()  # 模型进入训练模式
            start = time.time()
            for X, y in self.train_dataloader:
                # 数据搬家
                X = X.to(device=self.device)
                y = y.to(device=self.device)

                # 正向传播
                y_pred = self.model(X)
                # 计算误差
                loss = self.loss_fn(y_pred, y)
                # 反向梯度下降
                loss.backward()
                # 优化器
                self.optimizer.step()
                # 清空梯度
                self.optimizer.zero_grad()
            # 过程评估
            train_acc = self._get_acc(dataloader=self.train_dataloader)
            test_acc = self._get_acc(dataloader=self.test_dataloader)
            # 保存最好模型
            if test_acc > cur_acc:
                cur_acc = test_acc
                best_name = os.path.join(ckpt, self.model_type + "best.pt")
                torch.save(obj=self.model.state_dict(), f=best_name)

            # 保存last模型（最后一轮的模型）
            last_name = os.path.join(ckpt, self.model_type + "last.pt")
            torch.save(obj=self.model.state_dict(), f=last_name)

            stop = time.time()
            print(f"Eooch: {epoch + 1}, Train_Acc: {train_acc}, Test_Acc: {test_acc}, Time: {stop - start}")


def predict(model_type=None, img_file=None):
    """
        predict(img_file="./gestures/test/G5/IMG_1204.JPG")
    :return:
    """
    if img_file == None:
        return 'img_file=None'

    if model_type == "LeNet5":
        model_type = "LeNet_5_"

        # 加载模型
        model = LeNet_5()
        model.load_state_dict(state_dict=torch.load(T.get_file_path("models", model_type + "best.pt")))
    elif model_type == "VGG16":
        model_type = "VGG_16_"

        # 加载模型
        model = VGG_16()
        model.load_state_dict(state_dict=torch.load(T.get_file_path("models", model_type + "best.pt")))
    elif model_type == "ResNet50":
        model_type = "ResNet_50_"

        # 加载模型
        model = ResNet_50()
        model.load_state_dict(state_dict=torch.load(T.get_file_path("models", model_type + "best.pt")))
    else:
        model_type = "LeNet_5_"

        # 加载模型
        model = LeNet_5()
        model.load_state_dict(state_dict=torch.load(T.get_file_path("models", model_type + "best.pt")))

    if model_type:
        pass
    else:
        model_type = "LeNet_5_"

        # 加载模型
        model = LeNet_5()
        model.load_state_dict(state_dict=torch.load(T.get_file_path("models", model_type + "best.pt")))

    # 打开图像
    img = Image.open(img_file)
    # 预处理
    img = MyDataSet.my_preprocess(img)
    img = torch.unsqueeze(input=img, dim=0)

    # 模型推理
    model.eval()
    with torch.no_grad():
        y_pred = model(img)
        y_pred = torch.argmax(input=y_pred, dim=1).item()

    # 输出结果
    return f"G{y_pred}"
