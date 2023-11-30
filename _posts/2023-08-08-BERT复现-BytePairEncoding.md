---
title: BERT复现-详解BytePairEncoding算法
date: 2023-08-08 10:00:32
tags:
---

# Byte Pair Encoding
算法流程：
1. 设定最大subwords个数 
2. 将所有单词拆分为单个字符，并在最后添加一个停止符 </w>，同时标记出该单词出现的次数。例如，"low" 这个单词出现了 5 次，那么它将会被处理为 {'l o w </w>': 5}
3. 统计每一个连续字节对的出现频率，选择最高频者合并成新的 subword, 如统计出来 l o最高频，则会将{'l o w </w>': 5}替换为{'lo w </w>': 5}（去掉中间的空格）
4. 重复第3步直到达到第1步设定的subwords词表大小或下一个最高频的字节对出现频率为1
5. 遍历词表，统计各个subword，输出

```python
import re, collections

# 步骤2，将所有单词拆分，添加停止符并统计出现次数
def get_vocab(filename):
    vocab = collections.defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as fhand:
        for line in fhand:
            words = line.strip().split()
            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

# 步骤3，统计连续字节对的出现频率
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_tokens(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens

# vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}

# Get free book from Gutenberg
# wget http://www.gutenberg.org/cache/epub/16457/pg16457.txt
vocab = get_vocab('pg16457.txt')

print('==========')
print('Tokens Before BPE')
tokens = get_tokens(vocab)
print('Tokens: {}'.format(tokens))
print('Number of tokens: {}'.format(len(tokens)))
print('==========')

num_merges = 1000
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print('Iter: {}'.format(i))
    print('Best pair: {}'.format(best))
    tokens = get_tokens(vocab)
    print('Tokens: {}'.format(tokens))
    print('Number of tokens: {}'.format(len(tokens)))
    print('==========')
```


参考：https://leimao.github.io/blog/Byte-Pair-Encoding/
