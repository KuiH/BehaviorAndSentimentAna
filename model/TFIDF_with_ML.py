import os
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='[%(asctime)s][%(name)s][%(levelname)s]%(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str):
    return text.replace("，，，", "，").replace("，，", ",") # 简单处理一下


def load_and_clean_data(data_dir: str):
    df_train = pd.read_csv(os.path.join(data_dir, "comments_small_train.csv"))
    df_val = pd.read_csv(os.path.join(data_dir, "comments_small_val.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "comments_small_test.csv"))
    df_train['comment'] = df_train['comment'].apply(clean_text)
    df_val['comment'] = df_val['comment'].apply(clean_text)
    df_test['comment'] = df_test['comment'].apply(clean_text)
    logger.info(df_train.head())
    return df_train, df_val, df_test


def tfidf_vector(train_text:pd.Series, val_text:pd.Series, test_text:pd.Series):
    """使用tf-idf的方式进行向量化"""
    # 使用最常出现的6000词作为特征词汇
    vectorizer = TfidfVectorizer(max_features=6000)
    X_train_tfidf = vectorizer.fit_transform(train_text)
    X_val_tfidf = vectorizer.transform(val_text)
    X_test_tfidf = vectorizer.transform(test_text)
    return X_train_tfidf, X_val_tfidf, X_test_tfidf





