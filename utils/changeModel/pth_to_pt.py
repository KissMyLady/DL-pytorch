# coding:utf-8
# Author:mylady
# Datetime:2023/3/22 22:34
import torch
# from models.network_swinir import SwinIR #你训练的网络

import torchvision


# https://blog.csdn.net/weixin_49379314/article/details/126177648
def test_1():
    model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                   img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                   mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')  # 训练此网络时往里输入的参数

    state_dict = torch.load("S_x4.pth")  # 要转换的文件路径
    model.load_state_dict(state_dict, False)
    model.eval()  # 切换到eval（）

    example = torch.rand(1, 3, 320, 480)  # 生成一个随机输入维度的输入
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("model.pt")
    pass


def main():
    pass


if __name__ == '__main__':
    main()
