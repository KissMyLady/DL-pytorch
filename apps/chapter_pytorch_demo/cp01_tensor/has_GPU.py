import torch


def test_1():
    if torch.cuda.is_available():
        print("GPU是否可用: \t", torch.cuda.is_available())
        print("GPU数量: \t", torch.cuda.device_count())
        print("GPU索引号: \t", torch.cuda.current_device())
        print("GPU名称: \t", torch.cuda.get_device_name())

    else:
        print("warn: 当前服务器GPU不可用")


if __name__ == '__main__':
    test_1()
