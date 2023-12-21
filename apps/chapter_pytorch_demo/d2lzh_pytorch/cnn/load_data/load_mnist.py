# coding:utf-8
# Author:mylady
# Datetime:2023/7/01 17:11

import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from d2lzh_pytorch.myUtils import Timer



def get_dataloader_workers():
    """
        Use 4 processes to read the data.
    """
    return 4


# ~/Datasets/FashionMNIST
# /mnt/g1t/ai_data/Datasets_on_HHD
def load_data_fashion_mnist(batch_size,
                            resize=None,
                            root='/home/mylady/ai_data/d2l_data',
                            download=False
                            ):
    """
        Download the Fashion-MNIST dataset and then load it into memory.
    """
    timer = Timer()
    timer.start()

    # 图像增广
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)
    # 组装
    mnist_train = torchvision.datasets.FashionMNIST(root=root, 
                                                    train=True, 
                                                    transform=trans, 
                                                    download=download)
    mnist_test = torchvision.datasets.FashionMNIST(root=root,
                                                    train=False,
                                                    transform=trans, 
                                                    download=download)
    # 返回数据
    res_1 = DataLoader(mnist_train, 
                       batch_size, 
                       shuffle=True,
                       num_workers=get_dataloader_workers()
                       )
    res_2 = DataLoader(mnist_test, 
                       batch_size, 
                       shuffle=False,
                       num_workers=get_dataloader_workers()
                       )
    s_time = timer.stop()
    print("read Fashion-MNIST Dataset consume time %.2f s" % s_time)
    return (res_1, res_2)


def test_1():
    
    pass


def test_2():
    lr = 0.1
    num_epochs = 10
    batch_size = 128

    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    # 训练
    train_ch6(net, train_iter, test_iter, 
              num_epochs, 
              lr, 
              myUtils.try_gpu()
             )
    pass


def main():
    pass


if __name__ == "__main__":
    # main()
    pass