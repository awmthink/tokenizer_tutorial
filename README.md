# Tokenizer

关于Tokenizer系统性介绍的文档：https://zazt7evze69.feishu.cn/wiki/Syr6wQFRIilZfJkQqu6cIw2Inmb

本教程中相关notebook和代码的功能说明：

* [basic_usage.ipynb](./basic_usage.ipynb): 介绍了transformers中tokenizer模块的基本的用法
* [tokenizers_lib.ipynb](./tokenizers_lib.ipynb): 介绍了HuggingFace的tokenizers库的用法，包括了它的Normalizer, Pre-Tokenizer，Models，Postprocessor，Trainers等。
* [bpe_from_scratch.py](./bpe_from_scratch.py): 纯 Python 代码演示了 BPE 算法的原理与实现
* [wordpiece_from_scratch.py](./wordpiece_from_scratch.py): 纯 Python 代码演示了 WordPiece 算法的原理与实现
* [unigram_from_scratch.py](./unigram_from_scratch.py): 纯P ython 代码演示了 unigram 算法的原理与实现
* [build_domain_tokenzier.py](./build_domain_tokenizer.py): 该脚本是对于 SentencePiece 中`SentencePieceTrainer.train`的包装，用于训练一个新的 Tokenizer。
* [merge_tokenizers.py](./merge_tokenizers.py)：该脚本用于Tokenizer的合并，它在一个base tokenizer的基础上扩充新训练的特定领域的tokenizer，进一步扩充了百川的词汇表和结巴分词中的高频词汇。
* [spm_extractor.py](./spm_extractor.py): 该脚本用于将SenetencePiece工具训练的BPE模型的`.model`文件转换为`vocab.json`和`merges.txt`。
* [Tokenizer Viewer](./tokenizer_viewer_webui.py)：基于 Streamlit 写的一个 tokenizer 的可视化工具，可以对比多个 tokenizer 词汇表的差别。