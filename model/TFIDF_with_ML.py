import os
import pandas as pd
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import logging

from utils import load_stopwords, tokenize_cn_text, clean_text

logging.basicConfig(format='[%(asctime)s][%(name)s][%(levelname)s]%(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# 一些超参数
MAX_FEATURES = 64  # TF-IDF向量化的特证数
MAX_ITER = 100  # 模型迭代次数


def load_and_clean_data(data_dir: str):
    df_train = pd.read_csv(os.path.join(data_dir, "comments_small_train.csv"))
    df_val = pd.read_csv(os.path.join(data_dir, "comments_small_val.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "comments_small_test.csv"))
    df_train['cleaned_comment'] = df_train['comment'].apply(clean_text)
    df_val['cleaned_comment'] = df_val['comment'].apply(clean_text)
    df_test['cleaned_comment'] = df_test['comment'].apply(clean_text)
    return df_train, df_val, df_test


def tfidf_vector(train_text: pd.Series, val_text: pd.Series, test_text: pd.Series):
    """使用tf-idf的方式进行向量化"""
    logger.info("TF-IDF Vectorizing...")
    stopwords = load_stopwords('../resources/stopwords_hit.txt')  # 停用词文件路径

    def tokenize_and_filter_stopwords(text):
        tokens = tokenize_cn_text(text)
        return [t for t in tokens if t not in stopwords]

    # 使用最常出现的MAX_FEATURES词作为特征词汇
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, tokenizer=tokenize_and_filter_stopwords)
    X_train_tfidf = vectorizer.fit_transform(train_text)  # 用训练集构建vectorizer
    X_val_tfidf = vectorizer.transform(val_text)  # 验证集使用上面构建好的vectorizer
    X_test_tfidf = vectorizer.transform(test_text)
    # logger.debug(type(X_test_tfidf)) # <class 'scipy.sparse.csr.csr_matrix'>
    # logger.debug(pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out()).head())
    return X_train_tfidf, X_val_tfidf, X_test_tfidf


def do_test(model, tf_idf_vectors, mode: str):
    """
    :return: y_pred：预测类别; y_pred_proba: 各类别的概率
    """
    assert mode in ["validation", "test"]
    logger.info(f"Eval on {mode} set")
    y_pred = model.predict(tf_idf_vectors)
    y_pred_proba = model.predict_proba(tf_idf_vectors)
    return y_pred, y_pred_proba


def do_train(train_tfidf, train_label, val_tfidf=None, val_label=None):
    model = LogisticRegression(max_iter=MAX_ITER, class_weight="balanced")
    model.fit(train_tfidf, train_label)
    if val_tfidf is not None and val_label is not None:
        y_pred_val, _ = do_test(model, val_tfidf, mode="validation")
        logger.info("\n" + classification_report(val_label, y_pred_val))
    return model


def main():
    df_train, df_val, df_test = load_and_clean_data("../dataset")
    X_train, y_train = df_train["cleaned_comment"], df_train["label"]
    X_val, y_val = df_val["cleaned_comment"], df_val["label"]
    X_test, y_test = df_test["cleaned_comment"], df_test["label"]
    X_train_tfidf, X_val_tfidf, X_test_tfidf = tfidf_vector(X_train, X_val, X_test)

    model = do_train(X_train_tfidf, y_train, X_val_tfidf, y_val)
    y_pred_test, y_pred_proba_test = do_test(model, X_test_tfidf, mode="test")
    logger.info("\n" + classification_report(y_test, y_pred_test))


if __name__ == '__main__':
    main()
