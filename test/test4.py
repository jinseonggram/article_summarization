import json
import pandas
import numpy as np
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
import textwrap
import torch
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
from tqdm.auto import tqdm
# from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import PegasusTokenizer, BigBirdPegasusForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset

pl.seed_everything(42)
torch.cuda.empty_cache()


class NewsSummaryDataset(Dataset):

    def __init__(
            self,
            # data: pd.DataFrame,
            source_data,
            target_data,
            tokenizer,
            text_max_token_len: int = 1024,
            summary_max_token_len: int = 256
    ):
        self.tokenizer = tokenizer
        # self.data = data
        self.source_data = source_data
        self.target_data = target_data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, index: int):
        text_encoding = self.tokenizer(
            self.source_data,
            max_length=self.text_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        summary_encoding = self.tokenizer(
            self.target_data,
            max_length=self.summary_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        labels = summary_encoding['input_ids']
        labels[labels == 0] = -100

        return dict(
            text=self.source_data,
            summary_encoding=self.target_data,
            text_input_ids=text_encoding['input_ids'].flatten(),
            text_attention_mask=text_encoding['attention_mask'].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding["attention_mask"].flatten()
        )

class NewsSummaryDataModule(pl.LightningDataModule):

    def __init__(
            self,
            # train_df: pd.DataFrame,
            # test_df: pd.DataFrame,
            train,
            valid,
            tokenizer: BartTokenizer,
            batch_size: int = 8,
            text_max_token_len: int = 1024,
            summary_max_token_len: int = 256
    ):

        super().__init__()

        # self.train_df = train_df
        # self.test_df = test_df
        self.train = train
        self.valid = valid

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len


    def setup(self, stage=None):
        self.train_dataset = NewsSummaryDataset(
            # self.train_df,
            self.train['text'],
            self.train['summarization'],
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

        self.test_dataset = NewsSummaryDataset(
            # self.test_df,
            self.valid['text'],
            self.valid['summarization'],
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8
        )


MODEL_NAME = "sshleifer/distilbart-cnn-6-6"
toekenizer = BartTokenizer.from_pretrained(MODEL_NAME)

N_EPOCHS = 8
BATCH_SIZE = 4

# 데이터
raw_datasets = load_dataset('cnn_dailymail', '3.0.0')
raw_train = raw_datasets['train'][:30000]
raw_valid = raw_datasets['validation'][:5000]

train = {'text': [], 'summarization': []}
valid = {'text': [], 'summarization': []}

for text, summarization in zip(raw_train['article'], raw_train['highlights']):
    if len(toekenizer.encode(text)) < 1024 and len(toekenizer.encode(summarization)):
        train['text'].append(text)
        train['summarization'].append(text)


for text, summarization in zip(raw_valid['article'], raw_valid['highlights']):
    if len(toekenizer.encode(text)) < 1200 and len(toekenizer.encode(summarization)) < 300:
        valid['text'].append(text)
        valid['summarization'].append(text)

print('train set len :', len(train['text']), len(train['summarization']))
print('train set len :', len(valid['text']), len(valid['summarization']))


data_module = NewsSummaryDataModule(train, valid, toekenizer, batch_size=BATCH_SIZE)


class NewsSummaryModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        # self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        # self.model = BigBirdPegasusForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        self.model = BartForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):

        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.00001)

model = NewsSummaryModel()

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='best-checkpoint',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

logger = TensorBoardLogger('lightning_logs', name='news-summary')

trainer = pl.Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    accelerator='gpu',
    devices=[0,1,2,3],
    strategy='dp',
    # num_nodes=4,
    max_epochs=N_EPOCHS,
)


trainer.fit(model, data_module)