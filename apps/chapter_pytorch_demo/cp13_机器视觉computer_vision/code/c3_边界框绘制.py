# coding:utf-8
# Author:mylady
# Datetime:2023/3/21 12:32
from PIL import Image
import sys
sys.path.append("..")
# import d2lzh_pytorch as d2l
from apps.chapter_pytorch_demo import d2lzh_pytorch as d2l


# 画图函数
def bbox_to_rect(bbox, color):
    """
    左上x, 左上y, 右下x, 右下y
    """
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
    return d2l.plt.Rectangle(xy=(bbox[0], bbox[1]),
                             width=bbox[2]-bbox[0],
                             height=bbox[3]-bbox[1],
                             fill=False,
                             edgecolor=color,
                             linewidth=2
                            )


def bounding_box_print(img, dog_bbox=None, cat_bbox=None):
    d2l.set_figsize()
    d2l.plt.imshow(img)  # 加分号只显示图

    fig = d2l.plt.imshow(img)
    fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
    fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
    pass


def run():
    # 左, 上, 右, 下
    dog_bbox = [60, 45, 378, 516]
    cat_bbox = [400, 112, 655, 493]
    img = Image.open('../img/catdog.jpg')

    # 绘制
    bounding_box_print(img, dog_bbox, cat_bbox)
    pass


if __name__ == '__main__':
    run()
