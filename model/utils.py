import jieba
import re
from typing import *


def clean_text(text: str):
    cleaned_text = re.sub(r'\d+', '', text) # 去除数字
    return cleaned_text.replace("，，，", "，").replace("，，", ",")  # 简单处理一下逗号


def load_stopwords(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(f.read().strip().splitlines())


def tokenize_cn_text(text: str) -> List[str]:
    return jieba.lcut(text)
