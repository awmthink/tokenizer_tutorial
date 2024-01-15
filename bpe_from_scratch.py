# %%
from transformers import AutoTokenizer

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]


tokenizer = AutoTokenizer.from_pretrained("gpt2")

# %% 对语料库进行预分词，将语料库中每个句子分拆为一个个的word
# 预分词的有2个作用：
# 1）预分词后的词汇之间不会产生合并；2）按词汇的粒度来统计频率，比从原始句子中统计频率效率要高

from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
        text
    )
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)
# %% 构建词汇表，这里使用语料库中出现的基本字符作为初始词汇

alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()
vocab = ["<|endoftext|>"] + alphabet.copy()

# %%
# splits保存了当前每个word在当前词汇表下拆分的subword
splits = {word: [c for c in word] for word in word_freqs.keys()}


# 计算每个word中相邻token pair出现的频率
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


# pair_freqs中存储了当前corpus中所有token pair出现的频率
pair_freqs = compute_pair_freqs(splits)


for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}: {pair_freqs[key]}")
    if i >= 5:
        break

# %% 找出频率最好的pair

best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq
print(best_pair, max_freq)
# %% 更新词表

merges = {best_pair: "".join(best_pair)}
vocab.append("".join(best_pair))


# %% 更新splits中的映射关系
def merge_pair(a, b, splits):
    for word in word_freqs.keys():
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits


splits = merge_pair(*best_pair, splits)
print(splits["Ġtrained"])

# %%
vocab_size = 50

# 以下的实现可以优化，比如在merge_pair时，有针对性的更新pair_freqs
# 当前现在的pair被合并时，将其从pair_freqs中减去，再添加新的merge pair
while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

print(vocab)
print(merges)


# %%
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split
        print(sum(splits, []))

    return sum(splits, [])


tokenize("This is not a token.")
