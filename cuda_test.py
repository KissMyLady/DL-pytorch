# coding:utf-8
# Author:mylady
# 2023/12/24 12:27
import torch

print(torch.__version__)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


if torch.cuda.is_available():
    print("GPU是否可用: \t", torch.cuda.is_available())

    print("GPU数量: \t", torch.cuda.device_count())

    print("GPU索引号: \t", torch.cuda.current_device())

    print("GPU名称: \t", torch.cuda.get_device_name())

    print("cuDNN是否可用, 打印版本号: \t", torch.backends.cudnn.version())  # 8700
else:
    print("warn: 当前服务器GPU不可用")


if __name__ == '__main__':
    pass