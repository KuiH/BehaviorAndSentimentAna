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
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 一些超参数
NUM_WORDS = 6000  # 使用出现频率最高的前n个词
MAX_SEQ_LEN = 60  # 最大输入长度
EMBEDDING_DIM = 128  # emb后的维度
LSTM_HIDDEN_DIM = 128  # LSTM隐含层维度
MLP_HIDDEN_DIM = 32  # 分类头(MLP)第一个线性层的输出维度
NUM_LAYERS = 1  # LSTM层数
# DROPOUT = 0.1


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
            return self.cache[idx]

        comment = self.comment_list[idx]
        label = self.label_list[idx]

        comment = clean_text(comment)
        token_ids = self.tokenizer.tokenize(comment)
        token_ids = torch.tensor(token_ids)
        label = torch.tensor(label, dtype=torch.long)
        self.cache[idx] = (token_ids, label)

        return token_ids, label


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, mlp_hidden_dim, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=num_layers,
                            bidirectional=False, batch_first=True)
        lstm_output_dim = lstm_hidden_dim

        # 分类头
        self.mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, text):
        # text = [batch_size, sent_len]

        embedded = self.embedding(text)  # [batch_size, sent_len, embedding_dim]

        lstm_out, (hidden, cell) = self.lstm(embedded) # lstm_out: [batch_size, sent_len, hidden_dim]
        # logger.debug(lstm_out.shape)

        # 用最后一个隐含层来分类
        hidden = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        output = self.mlp(hidden)  # [batch_size, 1]
        return output


if __name__ == '__main__':
    set_seed(SEED)

    # 下面是一些测试代码
    tokenizer = Tokenizer(vocab_path=r"D:\PythonProject\BehaviorAndSentimentAna\resources\vocab.json",
                          stop_words_path=r"D:\PythonProject\BehaviorAndSentimentAna\resources\stopwords_hit.txt",
                          num_words=NUM_WORDS, max_seq_len=MAX_SEQ_LEN)
    df_test = pd.read_csv(r"D:\PythonProject\BehaviorAndSentimentAna\dataset\comments_small_test.csv").head(8)
    dataset = CommentDataset(df_test, tokenizer)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    # logger.debug(dataset[0])
    # logger.debug(dataset[1])
    model = SentimentLSTM(vocab_size=NUM_WORDS, embedding_dim=EMBEDDING_DIM, lstm_hidden_dim=LSTM_HIDDEN_DIM,
                          mlp_hidden_dim=MLP_HIDDEN_DIM, num_layers=NUM_LAYERS)
    for tokens, labels in dataloader:
        model(tokens)
        break

