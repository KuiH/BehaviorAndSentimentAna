import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import *

from utils import clean_text, load_stopwords, tokenize_cn_text

logging.basicConfig(format='[%(asctime)s][%(name)s][%(levelname)s]%(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self, vocab_path: str, stop_words_path: str, num_words: int, max_seq_len: int):
        """
        :param vocab_path: 词表路径。词表中的词已经按照频率进行降序排序
        :param num_words: 保留的最大token数（去掉出现频率小的token）
        :param max_seq_len: 一个句子允许的最大长度（以token记）
        """
        self.vocab_size = num_words
        self.max_seq_len = max_seq_len
        self.stopwords = load_stopwords(stop_words_path)
        self.word_2_index = self.__load_vocab(vocab_path)
        self.index_2_word = {v: k for k, v in self.word_2_index.items()}  # 反向索引

        self.pad_token = "<pad>"
        self.pad_token_id = self.word_2_index[self.pad_token]
        self.unk_token = "<unk>"
        self.unk_token_id = self.word_2_index[self.unk_token]

    def __load_vocab(self, vocab_path: str):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        # 取前num_words词作为词表
        vocab = {token: index for i, (token, index) in enumerate(vocab.items()) if i < self.vocab_size}
        return vocab

    def tokenize(self, text: str) -> List[int]:
        tokens = tokenize_cn_text(text)
        tokens = [t for t in tokens if t not in self.stopwords]
        # token转index. 没见过的token对应unk
        token_ids = [self.word_2_index.get(token, self.unk_token_id) for token in tokens]
        if len(token_ids) > self.max_seq_len:  # 截断
            token_ids = token_ids[:self.max_seq_len]
        else:  # 填充
            token_ids += [self.pad_token_id] * (self.max_seq_len - len(token_ids))
        logger.debug(tokens)
        logger.debug(token_ids)
        return token_ids


if __name__ == '__main__':
    pass
