import sys
sys.path.append(".")
sys.path.append("../..")

import torch
from d2lzh_pytorch import myUtils


def predict_sentiment(net, vocab, sequence):
    """
    预测文本序列的情感
    """
    sequence = torch.tensor(vocab[sequence.split()], device=myUtils.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'


# 使用测试类的数据进行测序
def test_1():
  # data_dir = d2l.download_extract('aclImdb', 'aclImdb')
  from d2lzh_pytorch.nlp.load_data.load_imdb import read_imdb
  data_dir = "/mnt/g1t/ai_data/Datasets_on_HHD/data/aclImdb"

  test_data = read_imdb(data_dir, is_train=False)
  print('测试集数目：', len(test_data[0]))

  net = net.to("cuda")
  sum_total = 0
  acc_sum = 0
  for x, y in zip(test_data[0], test_data[1]):      
      predict_res = predict_sentiment(net, vocab, x)
      label = 'positive' if y == 1 else 'negative'
      if label == predict_res:
          # 预测正确
          acc_sum += 1
          pass
      else:
          # 预测错误
          print('测试数据集 标签错误：', y, '内容review:', x[0:60], '预测结果: ', predict_res)
          pass

      sum_total += 1
      if sum_total >= 100:
          break

  print('acc: %.4f' % (acc_sum / sum_total))
  pass