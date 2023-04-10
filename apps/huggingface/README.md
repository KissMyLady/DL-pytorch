Huggingface 工具集快速使用入门 以及 中文任务示例

<br>


### 启用代理
```sh
nohup jupyter notebook --notebook-dir=[dir] --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.password=sha1:xxxx &

# 在当前启动shell设置代理
export https_proxy=127.0.0.1:7890
export http_proxy=127.0.0.1:7890

jupyter-lab --allow-root --ip=0.0.0.0 --port=8888
```

## 依赖
```sh
torch
torchvision
torchaudio
torchtext

transformers
dataset

scikit-learn
```