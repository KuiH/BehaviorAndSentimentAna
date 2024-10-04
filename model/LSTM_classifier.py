import os
import json
import logging
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
from typing import *

from utils import clean_text, load_stopwords, tokenize_cn_text

logging.basicConfig(format='[%(asctime)s][%(name)s][%(levelname)s]%(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 一些超参数(用下面的参数，使用显存不超过2G)
NUM_WORDS = 6000  # 使用出现频率最高的前n个词
MAX_SEQ_LEN = 60  # 最大输入长度
EMBEDDING_DIM = 128  # emb后的维度
LSTM_HIDDEN_DIM = 128  # LSTM隐含层维度
MLP_HIDDEN_DIM = 32  # 分类头(MLP)第一个线性层的输出维度
NUM_LAYERS = 1  # LSTM层数
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.001


# DROPOUT = 0.1 不用dropout


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

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # pad_token_id = 0
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
        embedded = self.embedding(text)  # [batch_size, sent_len, embedding_dim]

        lstm_out, (hidden, cell) = self.lstm(embedded)  # lstm_out: [batch_size, sent_len, hidden_dim]
        # logger.debug(lstm_out.shape)

        # 用最后一个隐含层来分类
        hidden = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        output = self.mlp(hidden)  # [batch_size, 1]
        output = torch.sigmoid(output)
        return output


def evaluate_model_on_test_set(model, test_dataloader, thresholds=np.arange(0.0, 1.02, 0.02)):
    """
    Function to evaluate the model on the test set with different thresholds, plot F1-score vs. threshold,
    and logger.info classification report for the best threshold.

    :param model: Trained LSTM model.
    :param test_dataloader: Dataloader for the test dataset.
    :param thresholds: Array of threshold values to test.
    """

    # Load the best model
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    model.eval()  # Set model to evaluation mode

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, total=len(test_dataloader), desc=f"Test: "):
            tokens, labels = batch
            tokens = tokens.to(DEVICE)
            labels = labels.to(DEVICE)

            labels = labels.float()
            predictions = model(tokens).squeeze(1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # List to store F1-scores for different thresholds
    f1_scores = []

    for threshold in thresholds:
        binary_predictions = (all_predictions > threshold).astype(int)
        f1 = f1_score(all_labels, binary_predictions)
        f1_scores.append(f1)

    # Find the threshold that gives the best F1-score
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]

    # # 保存图片，输出最佳结果
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores, marker='o', linestyle='-', color='b')
    plt.title('F1-score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1-score')
    plt.grid(True)

    save_path = "pics/f1_score_vs_threshold.png"
    plt.savefig(save_path)
    plt.close()

    logger.info(f"F1-score vs Threshold plot saved to {save_path}")
    logger.info(f"Best Threshold: {best_threshold}")

    best_binary_predictions = (all_predictions > best_threshold).astype(int)

    report = classification_report(all_labels, best_binary_predictions, target_names=['Negative', 'Positive'])
    logger.info("Classification report for the best threshold:")
    logger.info(f"\n{report}")

    # 阈值0.5也输出一下
    best_binary_predictions = (all_predictions > 0.5).astype(int)

    report = classification_report(all_labels, best_binary_predictions, target_names=['Negative', 'Positive'])
    logger.info("Classification report for threshold 0.5:")
    logger.info(f"\n{report}")


def main(do_train: bool = True, do_test: bool = True):
    model = SentimentLSTM(vocab_size=NUM_WORDS, embedding_dim=EMBEDDING_DIM, lstm_hidden_dim=LSTM_HIDDEN_DIM,
                          mlp_hidden_dim=MLP_HIDDEN_DIM, num_layers=NUM_LAYERS)
    model.to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Early stopping 参数
    early_stop_patience = 3  # Number of epochs to wait before stopping if no improvement
    best_valid_loss = float('inf')
    early_stop_counter = 0

    # 保存loss，以便画图
    train_losses = []
    valid_losses = []

    # 加载数据
    tokenizer = Tokenizer(vocab_path=r"D:\PythonProject\BehaviorAndSentimentAna\resources\vocab.json",
                          stop_words_path=r"D:\PythonProject\BehaviorAndSentimentAna\resources\stopwords_hit.txt",
                          num_words=NUM_WORDS, max_seq_len=MAX_SEQ_LEN)

    if do_train:
        df_train = pd.read_csv(r"D:\PythonProject\BehaviorAndSentimentAna\dataset\comments_small_train.csv")
        df_valid = pd.read_csv(r"D:\PythonProject\BehaviorAndSentimentAna\dataset\comments_small_val.csv")
        train_dataset = CommentDataset(df_train, tokenizer)
        valid_dataset = CommentDataset(df_valid, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

        # Training loop
        for epoch in range(EPOCHS):
            model.train()  # Set the model to training mode

            epoch_train_loss = 0
            for batch in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Training Epoch {epoch + 1}: "):
                tokens, labels = batch
                tokens = tokens.to(DEVICE)
                labels = labels.to(DEVICE)

                labels = labels.float()
                optimizer.zero_grad()

                predictions = model(tokens).squeeze(1)
                loss = criterion(predictions, labels)

                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)

            # 一个epoch结束，在验证集上跑一遍
            model.eval()
            epoch_valid_loss = 0

            with torch.no_grad():
                for batch in tqdm(valid_dataloader, total=len(valid_dataloader), desc=f"Eval: "):
                    tokens, labels = batch
                    tokens = tokens.to(DEVICE)
                    labels = labels.to(DEVICE)

                    labels = labels.float()
                    predictions = model(tokens).squeeze(1)
                    loss = criterion(predictions, labels)  # Calculate validation loss

                    epoch_valid_loss += loss.item()

            avg_valid_loss = epoch_valid_loss / len(valid_dataloader)
            valid_losses.append(avg_valid_loss)

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                torch.save(model.state_dict(), 'checkpoints/best_model.pth')
                early_stop_counter = 0  # Reset early stopping counter
            else:
                early_stop_counter += 1

            # Early stopping
            if early_stop_counter >= early_stop_patience:
                logger.info(f'Early stopping triggered after epoch {epoch + 1}.')
                break

            # Print training and validation loss for the epoch
            logger.info(
                f'Epoch {epoch + 1}/{EPOCHS}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}')

            # Plot validation loss after each epoch
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue', linestyle='-')
            plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss', color='orange',
                     linestyle='--')

            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('pics/loss_curve.png')
            plt.close()

    if do_test:
        logger.info("Do test...")
        df_test = pd.read_csv(r"D:\PythonProject\BehaviorAndSentimentAna\dataset\comments_small_test.csv")
        test_dataset = CommentDataset(df_test, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        evaluate_model_on_test_set(model, test_dataloader=test_dataloader)


if __name__ == '__main__':
    set_seed(SEED)
    main(do_train=False, do_test=True)

    # # 下面是一些测试代码
    # tokenizer = Tokenizer(vocab_path=r"D:\PythonProject\BehaviorAndSentimentAna\resources\vocab.json",
    #                       stop_words_path=r"D:\PythonProject\BehaviorAndSentimentAna\resources\stopwords_hit.txt",
    #                       num_words=NUM_WORDS, max_seq_len=MAX_SEQ_LEN)
    # df_test = pd.read_csv(r"D:\PythonProject\BehaviorAndSentimentAna\dataset\comments_small_test.csv").head(8)
    # dataset = CommentDataset(df_test, tokenizer)
    # dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    #
    # # logger.debug(dataset[0])
    # # logger.debug(dataset[1])
    # model = SentimentLSTM(vocab_size=NUM_WORDS, embedding_dim=EMBEDDING_DIM, lstm_hidden_dim=LSTM_HIDDEN_DIM,
    #                       mlp_hidden_dim=MLP_HIDDEN_DIM, num_layers=NUM_LAYERS)
    # for tokens, labels in dataloader:
    #     model(tokens)
    #     break
