import os
import json
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import *

from utils import clean_text, load_stopwords, tokenize_cn_text

logging.basicConfig(format='[%(asctime)s][%(name)s][%(levelname)s]%(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if DEVICE.type != "cpu" and torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        return token_ids


class CommentDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, tokenizer: Tokenizer):
        """
        :param data_df: 评论和标签的DataFrame
        """
        self.comment_list = data_df['comment'].tolist()
        self.label_list = data_df['label'].tolist()
        assert len(self.comment_list) == len(self.label_list)

        self.tokenizer = tokenizer
        self.cache: List[Tuple] = [None] * len(self.label_list)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        # 先查缓存
        if self.cache[idx] is not None:
            logger.debug("From cache")
            return self.cache[idx]

        comment = self.comment_list[idx]
        label = self.label_list[idx]

        comment = clean_text(comment)
        token_ids = self.tokenizer.tokenize(comment)
        token_ids = torch.tensor(token_ids)
        label = torch.tensor(label, dtype=torch.long)
        self.cache[idx] = (token_ids, label)

        return token_ids, label


if __name__ == '__main__':
    set_seed(SEED)
    tokenizer = Tokenizer(vocab_path=r"D:\PythonProject\BehaviorAndSentimentAna\resources\vocab.json",
                          stop_words_path=r"D:\PythonProject\BehaviorAndSentimentAna\resources\stopwords_hit.txt",
                          num_words=6000, max_seq_len=30)
    df_test = pd.read_csv(r"D:\PythonProject\BehaviorAndSentimentAna\dataset\comments_small_test.csv").head(8)
    dataset = CommentDataset(df_test, tokenizer)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    for tokens, labels in dataloader:
        print(tokens)
        print(labels)
