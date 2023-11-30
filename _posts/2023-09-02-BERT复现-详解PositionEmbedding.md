---
title: BERT复现-详解PositionEmbedding
date: 2023-09-02 10:30:12
tags:
---

talk is cheap, show me the code

首先看transformer原文中的定义：
![](/assets/image/BERT/embedding.png)

PE是一个(max_len, d_model)的Tensor，其中每个位置的值由如上公式定义，其中pos表示单词的位置，i是纬度。
代码实现上来说，先构建position，令其为torch.arange(0, max_len)的序列，然后增加维度。再构建分母部分的div_term, div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()。
二者相乘后分别取sin、cos值进行填入即可


```python
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
```

