from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import PegasusTokenizer
import csv


class NewsSummaryDataset(Dataset):
    def __init__(
            self,
            data,
            tokenizer: PegasusTokenizer,
            text_max_token_len: int = 512,
            summary_max_token_len: int = 128
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

        self.source = []
        self.target = []

        src_path = self.data + '.src.csv'
        trg_path = self.data + '.trg.csv'

        with open(src_path, 'r', encoding='utf-8') as src_file, open(src_path, 'r', encoding='utf-8') as trg_file:
            rdr_src = csv.reader(src_file)
            rdr_trg = csv.reader(trg_path)
            for src_line, trg_line in zip(rdr_src, rdr_trg):
                self.source.append(src_line)
                self.target.append(trg_line)

            print('source dataset 1, ')
            print(self.source[1])
            print('target dataset 1, ')
            print(self.target[1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        text_encoding = self.tokenizer(
            self.source,
            max_length=self.text_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        summary_encoding = self.tokenizer(
            self.target,
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
            text=self.source,
            summary_encoding=self.target,
            text_input_ids=text_encoding['input_ids'].flatten(),
            text_attention_mask=text_encoding['attention_mask'].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding["attention_mask"].flatten()
        )


class NewsSummaryDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train,
            test,
            model_name,
            batch_size: int = 64,
            text_max_token_len: int = 4096,
            summary_max_token_len: int = 1024
    ):

        super().__init__()

        self.train = train
        self.test = test
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def setup(self, stage=None):

        self.train_dataset = NewsSummaryDataset(
            self.train,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

        self.test_dataset = NewsSummaryDataset(
            self.test,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )