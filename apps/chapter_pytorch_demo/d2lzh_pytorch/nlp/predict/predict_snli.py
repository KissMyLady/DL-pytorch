
import torch
from d2lzh_pytorch import myUtils



def predict_snli(net, vocab, premise, hypothesis):
    """预测前提和假设之间的逻辑关系
    
    entailment: 限定继承
    contradiction: 矛盾
    neutral: 自然
    """
    net.eval()
    premise = torch.tensor(vocab[premise], device=myUtils.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=myUtils.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)), 
                              hypothesis.reshape((1, -1))]
                            ), dim=1)
    
    return 'entailment' if label == 0 else 'contradiction' if label == 1 else 'neutral'



def main():
    pass


if __name__ == "__main__":
    # main()
    pass