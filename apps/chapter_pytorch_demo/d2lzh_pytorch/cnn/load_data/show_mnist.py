import torch
from matplotlib import pyplot as plt


def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# 显示minst数据集
def test_1():
    import torchvision
    from torchvision import transforms
    from torch.utils.data.dataloader import DataLoader

    trans = [transforms.ToTensor()]
    #if resize:
    #    trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    # 组装
    rootPath='/mnt/aiguo/ai_data/Datasets_on_HHD/FashionMNIST'
    mnist_train = torchvision.datasets.FashionMNIST(root=rootPath, 
                                                    train=True, 
                                                    transform=trans, 
                                                    download=False)
    #@tab pytorch
    X, y = next(iter(DataLoader(mnist_train, 
                                batch_size=18)))
    show_images(X.reshape(18, 28, 28), 
                2, 
                9, 
                titles=get_fashion_mnist_labels(y)
               )
    
    pass


def main():
    pass


if __name__ == "__main__":
    main()
    pass
