# coding:utf-8
# Author:mylady
# 2023/9/4 21:47
import evaluate


def test_3():
    # 评估指标计算——迭代计算
    accuracy = evaluate.load("accuracy")

    for ref, pred in zip([0, 1, 0, 1], [1, 0, 0, 1]):
        accuracy.add(references=ref, predictions=pred)

    print(accuracy.compute())

    # 迭代计算2
    accuracy = evaluate.load("accuracy")

    for refs, preds in zip([[0, 1], [0, 1]], [[1, 0], [0, 1]]):
        accuracy.add_batch(references=refs, predictions=preds)

    print(accuracy.compute())
    pass


def test_2():
    # 评估指标计算——全局计算
    accuracy = evaluate.load("accuracy")

    results = accuracy.compute(references=[0, 1, 2, 0, 1, 2],
                               predictions=[0, 1, 1, 2, 1, 0]
                               )

    print(results)

    pass


def test_1():
    # 查看支持的评估函数
    print(evaluate.list_evaluation_modules(include_community=False,
                                           with_details=True))

    # 加载评估函数
    accuracy = evaluate.load("accuracy")

    # 查看函数说明
    print(accuracy.description)
    print(accuracy.inputs_description)
    print(accuracy)

    pass


def main():
    test_1()
    pass


if __name__ == "__main__":
    main()
