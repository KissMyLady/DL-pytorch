# coding:utf-8
# Author:mylady
# Datetime:2023/4/20 22:06


def res_encoded_X():
    vocab_size = 10000
    num_hiddens = 768
    ffn_num_hiddens = 1024
    num_heads = 4

    norm_shape = [768]
    ffn_num_input = 768
    num_layers = 2
    dropout = 0.2

    encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, dropout)

    # 这里模拟表示两个句子token输入
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1, 1, 1]])

    # 模拟句子输入到编码器, 将句子编码
    encoded_X = encoder(tokens, segments, None)
    # 输出 [batch_size2 x 句子长度8 x num_hiddens 768] = [2, 8, 768]
    return encoded_X


def test_BERTEncoder():
    encoded_X = res_encoded_X()

    print(encoded_X.shape)
    print('将句子编码后的输出: \n', encoded_X)
    pass


def test_MaskLM():
    vocab_size = 10000
    num_hiddens = 768
    encoded_X = res_encoded_X()

    # 实例化
    mlm = MaskLM(vocab_size, num_hiddens)

    # 输入三个 对应预测位置
    mlm_positions = torch.tensor([[1, 5, 2],
                                  [6, 1, 5]])

    # 对于每个预测
    mlm_Y_hat = mlm(encoded_X, mlm_positions)

    print(mlm_Y_hat.shape)
    print(mlm_Y_hat)

    # 损失计算
    mlm_Y = torch.tensor([[7, 8, 9],
                          [10, 20, 30]])
    loss = nn.CrossEntropyLoss(reduction='none')
    mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))

    print('loss计算: ', mlm_l.shape)
    print('mlm_l: \n', mlm_l)
    pass


def test_NextSentencePred():
    encoded_X = res_encoded_X()

    encoded_X = torch.flatten(encoded_X, start_dim=1)

    # NSP的输入形状:(batchsize，num_hiddens)
    nsp = NextSentencePred(encoded_X.shape[-1])

    nsp_Y_hat = nsp(encoded_X)

    print("nsp_Y_hat.shape: ", nsp_Y_hat.shape)
    print("nsp_Y_hat: \n", nsp_Y_hat)
    pass


def main():
    pass


if __name__ == '__main__':
    main()
