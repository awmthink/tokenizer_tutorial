#!/usr/bin/env python
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from transformers import AutoTokenizer, LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    return "\u4e00" <= uchar <= "\u9fa5"


def is_chinese_string(string):
    """判断是否全为汉字"""
    return all(is_chinese(c) for c in string)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "llama2_tokenizer",
        type=str,
        metavar="NAME_OR_PATH",
        help="Llama2模型Tokenizer的路径",
    )
    parser.add_argument(
        "zh_sp_model",
        type=str,
        metavar="PATH",
        help="要合入Llama Tokeizer的中文SentencePiece模型",
    )
    parser.add_argument(
        "--output_tokenizer_path",
        default="./merged_zh_tokenizer",
        type=str,
        metavar="PATH",
        help="指定合并后的Tokenizer的写入路径",
    )

    args = parser.parse_args()

    llama_tokenizer = AutoTokenizer.from_pretrained(
        args.llama2_tokenizer, trust_remote_code=True, use_fast=False
    )
    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    print(f"Load Llama Tokenizer, vocab size: {len(llama_tokenizer)}")

    zh_sp_model = spm.SentencePieceProcessor()
    zh_sp_model.Load(args.zh_sp_model)
    print(f"Load zh sentencepiece model, vocab size: {len(zh_sp_model)}")
    zh_spm = sp_pb2_model.ModelProto()
    zh_spm.ParseFromString(zh_sp_model.serialized_model_proto())

    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)

    added_set = set()
    llama_vocab_size = len(llama_spm.pieces)
    for p in zh_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set and is_chinese_string(piece):
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = -llama_vocab_size
            llama_spm.pieces.append(new_p)
            llama_vocab_size += 1
            added_set.add(piece)

    print(f"Added {len(added_set)} chinese tokens to Llama tokenizer vocab")

    # Save
    os.makedirs(args.output_tokenizer_path, exist_ok=True)
    sp_model_save_path = os.path.join(args.output_tokenizer_path, "chinese_llama.model")

    with open(sp_model_save_path, "wb") as f:
        f.write(llama_spm.SerializeToString())

    tokenizer = LlamaTokenizer(vocab_file=sp_model_save_path)
    tokenizer.save_pretrained(args.output_tokenizer_path)

    chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(args.output_tokenizer_path)

    text = "这是一段测试中文Tokenizer的文本。好雨知时节，当春乃发生。随风潜入夜，润物细无声。"
    print("Test text:\n", text)
    print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
    print(
        f"Tokenized by Chinese-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}"
    )


if __name__ == "__main__":
    main()
