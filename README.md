# 利用pytorch进行深度学习

说 明:
- 当前项目为 [d2l-en](https://github.com/d2l-ai/d2l-en) 的学习笔记, 使用了pytorch框架
- [d2l-zh项目地址](https://github.com/d2l-ai/d2l-zh) 中文版
- 包含huggingFace的学习代码
- 功 能: 
- - 1, 中英文互相翻译
- - 2, 阅读理解
- - 3, 句子填空测试
- - 4, 句子情感判断


DL-PyTorch书籍 
- [https://github.com/ShusenTang/Dive-into-DL-PyTorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch)

pytorch官方文档
- [https://pytorch.org/docs/](https://pytorch.org/docs/)

zh-d2l.ai书籍 在线阅读地址: 
- [http://zh.d2l.ai/chapter_introduction/index.html](http://zh.d2l.ai/chapter_introduction/index.html)


书籍下载地址: 
- [https://zh-v2.d2l.ai/d2l-zh-pytorch.pdf](https://zh-v2.d2l.ai/d2l-zh-pytorch.pdf)

## conda环境安装
```sh
# 创建
$ conda create -n d2l_pytorch python=3.10.*

# 激活
$ conda activate d2l_pytorch

# 退出
$ conda deactivate

# 删除
$ conda remove --name <env_name> --all
```


## 常用命令

- 启动jupyter命令
```shell
jupyter notebook --config /home/mylady/.jupyter/jupyter_notebook_config.py

nohup jupyter notebook --notebook-dir=[dir] --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.password=sha1:xxxx &


# on windows
cd D:\code\python\DL-pytorch
conda activate dl_pytorch
jupyter-lab --config  C:\Users\Administrator.SY-202304151755\.jupyter\jupyter_lab_config.py
```

## OpnCV4.0 中文文档

在线阅读
- [OpenCV 4.0 中文文档](https://opencv.apachecn.org/#/)
- [Opencv-doc-zh](https://github.com/apachecn/opencv-doc-zh)

