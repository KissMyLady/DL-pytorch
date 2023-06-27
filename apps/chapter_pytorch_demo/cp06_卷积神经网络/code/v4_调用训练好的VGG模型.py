
import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
sys.path.append("../..")
from d2lzh_pytorch.CNN.VGG_model import get_VGG_model
from d2lzh_pytorch.utils import train_ch5
from d2lzh_pytorch.utils import FlattenLayer
import d2lzh_pytorch.torch as d2l


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_ch3(net, test_iter, n=6):  # @save
    """预测标签（定义见第3章）"""

    err = 0
    succ = 0

    for X, y in test_iter:

        trues = d2l.get_fashion_mnist_labels(y)
        preds = d2l.get_fashion_mnist_labels(
            net(X).argmax(axis=1))  # 计算结果, 预测结果
        titles = [true + '\n' + pred for true, pred in zip(trues, preds)]

        X_resized = F.interpolate(X[0: n], size=(28, 28), mode='nearest')
        # print(X_resized.shape)   # 输出：torch.Size([6, 1, 28, 28])

        for true, pred in zip(trues, preds):
            if true != pred:
                # print("判断错误: 标签是:%s \t 计算为:%s" % (true, pred))
                err += 1
            else:
                succ += 1
                pass

        # d2l.show_images(X_resized.reshape((n, 28, 28)), 1, n, titles=titles[0:n])
        # break
        pass

    succ_per = succ / (succ + err)
    print("正确 %s, 错误: %s, 正确率: %s" % (succ, err, succ_per))
    # 正确 9134, 错误: 866, 正确率: 0.9134
    pass


def load_model_v2():
    # 加载已训练好的模型
    m_name = "../VGG_net_cpu_2023_06_27_11-53-28.pt"  # 7.8M
    model = torch.load(m_name)
    VGG = get_VGG_model()
    VGG.load_state_dict(model.state_dict())
    return VGG

def load_model_v2():
    # 加载已训练好的模型
    m_name = "../VGG_net_cpu_2023_06_27_11-53-28.pt"  # 7.8M
    VGG = torch.load(m_name)
    return VGG


def load_test_data():
    # # 加载数据
    batch_size = 256
    rootPath = r"/mnt/g1t/ai_data/Datasets_on_HHD/FashionMNIST"
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,
                                                        resize=224,
                                                        root=rootPath)
    return test_iter


def test_1():
    # 加载数据
    test_iter = load_test_data()

    # 加载模型
    VGG = load_model_v2()

    test_n = 6

    # 计算正确率
    predict_ch3(VGG,
                test_iter,
                n=test_n)
    pass


def main():
    test_1()
    pass


if __name__ == "__main__":
    main()
    pass
