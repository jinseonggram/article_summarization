import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar


class MyTrainer:
    def __init__(self, config):
        self.config = config
        self.checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints',
            filename='best-checkpoint',
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min'
        )
        self.logger = TensorBoardLogger('lightning_logs', name='news-summary')

    def train(self, model, data_module):
        trainer = pl.Trainer(
            logger=self.logger,
            callbacks=[self.checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
            accelerator='gpu',
            devices=1,
            max_epochs=self.config.n_epochs,
            num_sanity_val_steps=0
        )
        print('----- start training -----')
        trainer.fit(model, data_module)
        # return trainer
