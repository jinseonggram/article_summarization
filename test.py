from datasets import load_dataset
import json
from transformers import AutoTokenizer

raw_datasets = load_dataset('cnn_dailymail', '3.0.0')
model_ckpt = "sshleifer/distilbart-cnn-6-6"
toekenizer = AutoTokenizer.from_pretrained(model_ckpt)

# 데이터
# raw_datasets = load_dataset('cnn_dailymail', '3.0.0')
raw_train = raw_datasets['train'][:120000]
raw_valid = raw_datasets['validation'][:30000]

train = {'text': [], 'summarization': []}
valid = {'text': [], 'summarization': []}

for text, summarization in zip(raw_train['article'], raw_train['highlights']):
    if len(toekenizer.encode(text)) < 1024 and len(toekenizer.encode(summarization)):
        train['text'].append(text)
        train['summarization'].append(text)


for text, summarization in zip(raw_valid['article'], raw_valid['highlights']):
    if len(toekenizer.encode(text)) < 1024 and len(toekenizer.encode(summarization)):
        valid['text'].append(text)
        valid['summarization'].append(text)

print('train set len :', len(train['text']), len(train['summarization']))
print('train set len :', len(valid['text']), len(valid['summarization']))

# print(raw_datasets['train'][0].keys())  # dict_keys(['article', 'highlights', 'id'])
# print(raw_datasets['validation'][0].keys())
# print(raw_datasets['test'][0].keys())

# content_list = []
# sum_list = []
# a = 0
# data = raw_datasets['train'][:3]
# for c, s in zip(data['article'], data['highlights']):
#     # a += 1
#     # if a == 100000:
#     #     break
#     # text_encoding = tokenizer(
#     #     c,
#     #     max_length=1024,
#     #     padding="max_length",
#     #     truncation=True,
#     #     return_attention_mask=True,
#     #     add_special_tokens=True,
#     #     return_tensors="pt"
#     # )
#     #
#     # sum_encoding = tokenizer(
#     #     s,
#     #     max_length=1024,
#     #     padding="max_length",
#     #     truncation=True,
#     #     return_attention_mask=True,
#     #     add_special_tokens=True,
#     #     return_tensors="pt"
#     # )
#
#     text_token_count = len(tokenizer(c, max_length=None, truncation=False, return_tensors='pt'))
#     content_list.append(text_token_count)
#     summary_token_count = len(tokenizer(s, max_length=None, truncation=False, return_tensors='pt'))
#     sum_list.append(summary_token_count)
#
#     if len(tokenizer.encode(c)) > 1024:
#         print(tokenizer.encode(c))
#         print(tokenizer.encode(s))
#         print(len(tokenizer.encode(c)))
#         print(len(tokenizer.encode(s)))

# print(max(sum_list))
# print(sum(sum_list)/len(sum_list))
# print(len(sum_list))
#
# print(max(content_list))
# print(sum(content_list)/len(content_list))
# print(len(content_list))
