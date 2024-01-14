# Tokenizer

关于Tokenizer系统性介绍的文档：https://zazt7evze69.feishu.cn/wiki/Syr6wQFRIilZfJkQqu6cIw2Inmb

本教程中相关notebook和代码的功能说明：

* [basic_usage.ipynb](./basic_usage.ipynb): 介绍了transformers中tokenizer模块的基本的用法
* [tokenizers_lib.ipynb](./tokenizers_lib.ipynb): 介绍了HuggingFace的tokenizers库的用法，包括了它的Normalizer, Pre-Tokenizer，Models，Postprocessor，Trainers等。
* [bpe.py](./bpe.py): BPE算法的玩具级别的实现，演示了BPE算法训练与分词的原理。
* [build_domain_tokenzier.py](./build_domain_tokenizer.py): 该脚本是对于SentencePiece中`SentencePieceTrainer.train`的包装，用于训练一个新的Tokenizer。
* [merge_tokenizers.py](./merge_tokenizers.py)：该脚本用于Tokenizer的合并，它在一个base tokenizer的基础上扩充新训练的特定领域的tokenizer，进一步扩充了百川的词汇表和结巴分词中的高频词汇。
* [spm_extractor.py](./spm_extractor.py): 该脚本用于将SenetencePiece工具训练的BPE模型的`.model`文件转换为`vocab.json`和`merges.txt`。