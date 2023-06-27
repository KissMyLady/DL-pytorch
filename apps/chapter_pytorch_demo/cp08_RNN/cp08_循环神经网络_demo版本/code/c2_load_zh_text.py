# coding:utf-8
# Author:mylady
# Datetime:2023/3/23 13:54
import collections
import re
import jieba
from d2l import torch as d2l


# 加载汉字小说
def read_time_machine_v2():
    """
    将时间机器数据集加载到文本行的列表中
    """
    with open(r"X:\ai_data\NLP_data\贾平凹-山本.txt",
              'r', encoding='utf8') as f:
        lines = f.readlines().decode('utf-8')
        pass

    textList = jieba.cut(lines)
    q_cut_str = " ".join(textList).split()  # 先转为字符串，再按空格切分，返回列表

    # 中文分词处理
    # return [jieba.cut(line.strip().lower()) for line in lines]
    # return " ".join(lines).split()
    return q_cut_str


def main():
    # lines = read_time_machine()
    lines = read_time_machine_v2()

    print(f'# 文本总行数: {len(lines)}')
    print(lines[0])
    print(lines[10])
    pass


if __name__ == '__main__':
    main()
