# coding:utf-8
# Author:mylady
# Datetime:2023/4/9 11:50
import datasets
from datasets import load_from_disk


def main():
    dataset = datasets.load_dataset("lansinuote/ChnSentiCorp")
    dataset.save_to_disk('./data/ChnSentiCorp')

    dataset = load_from_disk("./data/ChnSentiCorp")
    pass


if __name__ == '__main__':
    main()
