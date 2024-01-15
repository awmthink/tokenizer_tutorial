from collections import defaultdict
from tokenizers import pre_tokenizers
from math import log
import copy

corpus = [  # The first sentences from the abstract of "<Attention Is All You Need>"
    "The dominant sequence transduction models are based on complex recurrent orconvolutional neural networks that include an encoder and a decoder.",
    "The bestperforming models also connect the encoder and decoder through an attentionmechanism.",
    "We propose a new simple network architecture, the Transformer,based solely on attention mechanisms, dispensing with recurrence and convolutionsentirely.",
]
#################### Step1: word freq ################
word_freqs = defaultdict(int)
pre_tokenizer = pre_tokenizers.Metaspace()

for text in corpus:
    words_with_offsets = pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, _ in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)
# defaultdict(<class 'int'>, {'▁The': 2, '▁dominant': 1, '▁sequence': 1, ...})

#################### Step2: initial vocab ################
char_freqs = defaultdict(int)  # 每个字符的频次
subwords_freqs = defaultdict(int)  # 每个 substring 的频次
for word, freq in word_freqs.items():
    for i in range(len(word)):
        char_freqs[word[i]] += freq
        # Loop through the subwords of length at least 2
        for j in range(i + 2, len(word) + 1):
            subwords_freqs[word[i:j]] += freq

sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
init_vocab_size = 300  # 一个较大的初始词表

# 基本字符 + 频率较高的一些subword 作为初始词表
token_freqs = (
    list(char_freqs.items()) + sorted_subwords[: init_vocab_size - len(char_freqs)]
)
token_freqs = {token: freq for token, freq in token_freqs}

print(sorted_subwords[:5])
# [('▁a', 12), ('an', 10), ('on', 10), ('en', 9), ('de', 9)]

#################### Step3: model ################
total_sum = sum([freq for token, freq in token_freqs.items()])
# model 存放每个候选 token 的负对数似然
model = {token: -log(freq * 1.0 / total_sum) for token, freq in token_freqs.items()}

#################### Step4: 定义编码函数和损失函数 ################


def encode_word(word, model):
    """这是用动态规划来实现维特比解码，从而根据每个子词的损失来分词"""
    best_segmentations = [{"start": 0, "score": 1}] + [
        {"start": None, "score": None} for _ in range(len(word))
    ]  # 核心数据结构，存放每个位置的状态：第 i 个元素表示对前缀 word[:i] 的分词结果：(最近一个拆分点, 最佳分词的损失)

    for start_idx in range(len(word)):
        # This should be properly filled by the previous steps of the loop
        best_score_at_start = best_segmentations[start_idx]["score"]  # 前缀的分词结果
        #########   寻找下一个拆分点   #############
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                if (
                    best_segmentations[end_idx]["score"] is None
                    or best_segmentations[end_idx]["score"] > score  # 损失更小
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}

    segmentation = best_segmentations[-1]  # 最后一个位置就是最终的分词结果
    if segmentation["score"] is None:
        # We did not find a tokenization of the word -> unknown
        return ["<unk>"], None

    score = segmentation["score"]
    start = segmentation["start"]  # 前一个拆分点
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score


def compute_loss(model):
    """计算当前语料库和模型的整体损失"""
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = encode_word(word, model)
        loss += freq * word_loss
    return loss


def compute_scores(model):
    """通过计算移除每个 token 的损失变化，从而计算每个 token 的得分"""
    scores = {}
    model_loss = compute_loss(model)
    for token, score in model.items():
        if len(token) == 1:  # 总是保留单个字符
            continue
        model_without_token = copy.deepcopy(model)
        _ = model_without_token.pop(token)
        scores[token] = compute_loss(model_without_token) - model_loss
    return scores


#################### Step5: 缩减词表 ################

percent_to_remove = 0.1  # 每轮迭代缩小 10%
max_vocab_size = 100  # 词表的最大规模
while len(model) > max_vocab_size:
    scores = compute_scores(model)
    # 按loss的变化从小到大排序，去除掉对loss变化影响最小的 p%
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    print("Top3 scores:%s" % sorted_scores[-3:])

    for i in range(int(len(model) * percent_to_remove)):  # 移除最小的 10%
        _ = token_freqs.pop(sorted_scores[i][0])

    ### 计算移除了一部分 token 后，整个词表的总频次，以及每个 token 出现的频率概率
    total_sum = sum([freq for token, freq in token_freqs.items()])
    model = {token: -log(freq * 1.0 / total_sum) for token, freq in token_freqs.items()}

# Top3 scores:[('ing', 8.45913446432769), ('form', 9.041467278547316), ('▁and', 9.270398846926355)]
# Top3 scores:[('form', 8.756385177048287), ('▁and', 8.84277569467804), ('tion', 9.158034534900253)]
# Top3 scores:[('rans', 11.55887624144998), ('▁The', 13.833700317065222), ('▁models', 21.35200333126363)]
# ...
