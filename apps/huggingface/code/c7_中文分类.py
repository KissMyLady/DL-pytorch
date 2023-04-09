# coding:utf-8
# Author:mylady
# Datetime:2023/4/9 9:07
import torch
from datasets import load_dataset
from transformers import BertTokenizer
from transformers import BertModel
from d2l import torch as d2l
from transformers import AdamW


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义数据集
class Dataset(torch.utils.data.Dataset):

    def __init__(self, split):
        dataset = load_dataset('lansinuote/ChnSentiCorp')
        print(dataset)

        def f(data):
            return len(data['text']) > 30

        self.dataset = dataset.filter(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        return text


# 定义数据集
dataset = Dataset('train')
len(dataset)  # , dataset[0]

# 加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')

# 加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese')

# 不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)


def collate_fn(data):
    # 编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=data,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=30,
                                   return_tensors='pt',
                                   return_length=True)
    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']

    # 把第15个词固定替换为mask
    labels = input_ids[:, 15].reshape(-1).clone()
    input_ids[:, 15] = token.get_vocab()[token.mask_token]
    # print(data['length'], data['length'].max())
    return input_ids, attention_mask, token_type_ids, labels


# 定义下游任务模型
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(768, token.vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(token.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
            pass

        out = self.decoder(out.last_hidden_state[:, 15])
        return out



def test_1():
    # 数据加载器
    loader = torch.utils.data.DataLoader(dataset=dataset['train'],
                                         batch_size=16,
                                         collate_fn=collate_fn,
                                         shuffle=True,
                                         drop_last=True

                                         )

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        print(len(loader))
        print(token.decode(input_ids[0]))
        print(token.decode(labels[0]))

        print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels.shape)
        break

    model = Model()

    # model(input_ids=input_ids,
    #       attention_mask=attention_mask,
    #       token_type_ids=token_type_ids
    #       ).shape
    #
    # model.to(device)

    from transformers import AdamW

    # 训练
    optimizer = AdamW(model.parameters(), lr=5e-4)

    # 损失函数
    loss = torch.nn.CrossEntropyLoss()

    model.train()

    # 开始训练
    for epoch in range(5):
        print("training on ", device)

        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
            out = model(input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device),
                        token_type_ids=token_type_ids.to(device)
                        )
            l = loss(out, labels)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    pass


def main():

    pass


if __name__ == '__main__':
    main()
