# coding:utf-8
# Author:mylady
# 2023/9/4 20:50
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")


def preprocess_function(example, tokenizer=tokenizer):
    model_inputs = tokenizer(example["content"],
                             max_length=512,
                             truncation=True)

    labels = tokenizer(example["title"],
                       max_length=32,
                       truncation=True)

    # label就是title编码的结果
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 函数映射
def test_3():
    # 重新构造
    processed_datasets = datasets.map(preprocess_function,
                                      num_proc=4,
                                      batched=True,
                                      # remove_columns=datasets["train"].column_names
                                      # remove_columns=['title', 'content']
                                      )

    processed_datasets

    pass

# 加载数据集中的某一项任务
def test_2():
    # glue项目有很多的子项目, 选择其中的boolq子项目
    boolq_dataset = load_dataset("super_glue", "boolq")
    print(boolq_dataset)
    pass


def test_1():
    # 加载数据集
    datasets = load_dataset("madao33/new-title-chinese")

    # 按照数据集划分进行加载
    dataset = load_dataset("madao33/new-title-chinese", split="train")

    dataset = load_dataset("madao33/new-title-chinese", split="train[10:100]")

    dataset = load_dataset("madao33/new-title-chinese", split="train[:50%]")

    dataset = load_dataset("madao33/new-title-chinese",
                           split=["train[:50%]", "train[50%:]"])

    # 查看数据集
    print(datasets["train"][0])

    print(datasets["train"][:2])

    ## 打印标题
    print(datasets["train"]["title"][:5])

    ## datasets["train"].column_names
    ## ['title', 'content']
    pass


def main():
    test_2()
    pass


if __name__ == "__main__":
    main()
