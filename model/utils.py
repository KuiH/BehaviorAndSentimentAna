import jieba
from typing import *

def clean_text(text: str):
    return text.replace("，，，", "，").replace("，，", ",") # 简单处理一下


def load_stopwords(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(f.read().strip().splitlines())


def tokenize_cn_text(text: str) -> List[str]:
    return jieba.lcut(text)
