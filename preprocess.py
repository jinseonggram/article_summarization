import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import argparse
import sys

csv.field_size_limit(sys.maxsize)
parser = argparse.ArgumentParser()

# 입력받을 인자값 등록
parser.add_argument('--path', required=True, help='data File path')


def preprocessing(args):
    f = open(args.path, 'r', encoding='utf-8')
    rdr = csv.reader(f)

    data = []
    content = []
    summary = []

    for line in rdr:
        if len(line) == 5:
            content.append(line[2])
            summary.append(line[3])
            data.append(line)
    f.close()

    X_train, X_test, y_train, y_test = train_test_split(content, summary, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

    print(len(y_val))
    print(len(y_test))
    print(len(y_train))

    save_csv('./data/train.trg.csv', y_train)
    save_csv('./data/train.src.csv', X_train)

    save_csv('./data/valid.trg.csv', y_val)
    save_csv('./data/valid.src.csv', X_val)

    save_csv('./data/test.trg.csv', y_test)
    save_csv('./data/test.src.csv', X_test)


def save_csv(path, text):
    f = open(path, 'w')
    writer = csv.writer(f, delimiter = ",", quotechar = '"', quoting = csv.QUOTE_ALL)
    writer.writerows([text])
    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    preprocessing(args)