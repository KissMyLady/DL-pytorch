import sys
sys.path.append("../..")
from d2lzh_pytorch import myUtils
from d2lzh_pytorch import myPolt

import torch
from torch import nn


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.

    Defined in :numref:`sec_use_gpu`"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, 
                                  segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    
    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), 
                 mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1, 1)
    
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    
    # 计算下一句子预测任务的损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    res_loss = mlm_l + nsp_l
    return mlm_l, nsp_l, res_loss


def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, myUtils.Timer()
    animator = myPolt.Animator(xlabel='step', 
                            ylabel='loss',
                            xlim=[1, num_steps], 
                            legend=['mlm', 'nsp'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = myUtils.Accumulator(4)
    num_steps_reached = False
    
    timer = myUtils.Timer()
    while step < num_steps and not num_steps_reached:
        
        for tokens_X, segments_X, valid_lens_x, \
             pred_positions_X, mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
        
            # 损失计算
            mlm_l, nsp_l, l = _get_batch_loss_bert(net, 
                                                   loss, 
                                                   vocab_size, 
                                                   tokens_X, 
                                                   segments_X, 
                                                   valid_lens_x,
                                                   pred_positions_X, 
                                                   mlm_weights_X, 
                                                   mlm_Y, 
                                                   nsp_y
                                                  )
            # 向前传播, 更新梯度
            l.backward()
            trainer.step()

            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1, (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            timer.stop()
            
            if step == num_steps:
                num_steps_reached = True
                break
            
            
    # print('耗时: ', timer)
    print('耗时: ', timer.sum())
    
    loss_1 = metric[0] / metric[3]
    loss_2 = metric[1] / metric[3]
    p_sec = metric[2] / timer.sum()
    
    print(f'MLM loss {loss_1:.3f}', f'NSP loss {loss_2:.3f}')
    print(f'{p_sec:.1f} sentence pairs/sec on ', f'{str(devices)}')


# 训练
def test_1():
    from d2lzh_pytorch.nlp.model.BERT_model import get_BERT_model
    from d2lzh_pytorch.nlp.load_data.load_wiki import load_data_wiki

    batch_size = 512
    max_len = 64
    # 加载数据
    train_iter, vocab = load_data_wiki(batch_size, max_len)

    # 模型
    net = get_BERT_model()

    # 损失函数
    loss = nn.CrossEntropyLoss()

    # GPU
    devices = try_all_gpus()

    num_steps = 50

    # 训练
    train_bert(train_iter,
               net, 
               loss, 
               len(vocab), 
               devices, 
               num_steps)    
    pass


def main():
    pass


if __name__ == "__main__":
    main()
    pass