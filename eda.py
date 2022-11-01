from transformers import (
    T5TokenizerFast as T5Tokenizer
)
from pylab import rcParams
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import argparse
import sys

csv.field_size_limit(sys.maxsize)
parser = argparse.ArgumentParser()

# 입력받을 인자값 등록
parser.add_argument('--path', required=True, help='data File path')

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
rcParams['figure.figsize'] = 16, 10

MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)


def tokenize(args):
    train_src = get_data(args.train_src_path)
    train_trg = get_data(args.train_trg_path)
    val_src = get_data(args.val_src_path)
    val_trg = get_data(args.val_trg_path)

    text_token_counts, summary_token_counts = [], []

    for i in range(len(train_trg)):
        text_token_count = len(tokenizer.encode(train_src[i]))
        text_token_counts.append(text_token_count)

        summary_token_count = len(tokenizer.encode(train_trg[i]))
        summary_token_counts.append(summary_token_counts)

    print(text_token_counts)
    print(summary_token_counts)


def visualize_eda(text_token_counts, summary_token_counts):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    sns.histplot(text_token_counts, ax=ax1)
    ax1.set_title("full text token counts")

    sns.histplot(summary_token_counts, ax=ax2)
    ax2.set_title("summary text token counts")


def get_data(path):
    f = open(path, 'r', encoding='utf-8')
    rdr = csv.reader(f)

    data = []

    for line in rdr:
        data.append(line)
    f.close()

    return data


def save_csv(path, text):
    f = open(path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerows([text])
    f.close()
