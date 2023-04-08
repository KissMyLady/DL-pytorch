# coding:utf-8
# Author:mylady
# Datetime:2023/4/9 3:03
from datasets import load_dataset
from datasets import load_from_disk

"""
数据集: https://huggingface.co/datasets/lansinuote/ChnSentiCorp


"""


# 数据读取与保存
def test_1():
    # 本地加载
    dataset = load_from_disk('../data/ChnSentiCorp')

    print(type(dataset))

    dataset.to_csv(path_or_buf='./data/ChnSentiCorp_test_v3.csv')
    pass


# 网络加载
def test_2():
    # 网络加载数据
    dataset = load_dataset("lansinuote/ChnSentiCorp")

    print(dataset)
    print(type(dataset))
    pass


def main():
    test_2()
    pass


if __name__ == '__main__':
    main()
