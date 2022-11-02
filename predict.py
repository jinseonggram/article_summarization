import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
from transformers import BartTokenizer, BartForConditionalGeneration
MODEL_NAME = "facebook/bart-large-cnn"
toekenizer = BartTokenizer.from_pretrained(MODEL_NAME)



class NewsSummaryModel(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        # self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
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
        return AdamW(self.parameters(), lr=0.0001)

trained_model = NewsSummaryModel.load_from_checkpoint(
    './model/best.ckpt'
)

trained_model.freeze()

def summarize(text):
    text_encoding = toekenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
               )

    generated_ids = trained_model.model.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    preds = [
        toekenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_id in generated_ids
    ]

    return "".join(preds)

# sample_row = test_df.iloc[0]
# text = sample_row['text']
text = "North Korea fired at least 10 missiles of various types from its east and west coasts on Wednesday, " \
       "South Korea’s Ministry of National Defense said. South Korea’s Joint Chiefs of Staff (JCS) said the launches " \
       "mark the first time a North Korean ballistic missile has fallen close to South Korea’s territorial waters – " \
       "south of the Northern Limit Line – since the division of Korea. The barrage of missile tests set off an air " \
       "raid warning in South Korea’s Ulleungdo island that sits about 120 kilometers (75 miiles) east of the Korean " \
       "Peninsula. JCS said one short-range ballistic missile fell in the international waters 167 kilometers (104 " \
       "miles) northwest of the island. Wednesday’s launch is North Korea’s 29th this year, according to a CNN count, " \
       "and comes after the United States and South Korea began previously scheduled military exercises called " \
       "“Vigilant Storm” on Tuesday. The maneuvers involve 240 aircraft and “thousands of service members” from both " \
       "countries, according to the US Defense Department. US Defense Secretary Lloyd Austin is scheduled to meet " \
       "with his South Korean counterpart Lee Jong-sup at the Pentagon on Thursday. "
model_summary = summarize(text)
print(model_summary)


