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

    print(type(dataset))  # <class 'datasets.dataset_dict.DatasetDict'>

    # dataset.to_csv(path_or_buf='./data/ChnSentiCorp_test_v3.csv')
    pass


# 网络加载
def test_2():
    # 网络加载数据
    dataset = load_dataset("lansinuote/ChnSentiCorp")

    print(dataset)
    print(type(dataset))  # <class 'datasets.dataset_dict.DatasetDict'>

    '''
    Found cached dataset parquet (C:/Users/mylady/.cache/huggingface/datasets/lansinuote___parquet/lansinuote--ChnSentiCorp-eaea6a9750cb0fe7/0.0.0/2a3b91fbd88a2c90d1dbbb32b460cf621d31bd5b05b934492fdef7d8d6f236ec)
    100%|██████████| 3/3 [00:00<00:00, 173.34it/s]
    DatasetDict({
        validation: Dataset({
            features: ['text', 'label'],
            num_rows: 1200
        })
        test: Dataset({
            features: ['text', 'label'],
            num_rows: 1200
        })
        train: Dataset({
            features: ['text', 'label'],
            num_rows: 9600
        })
    })
    '''
    pass


def main():
    test_1()
    pass


if __name__ == '__main__':
    main()
