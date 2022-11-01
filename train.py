import argparse
from model import NewsSummaryModel
from data_loader import NewsSummaryDataModule
from trainer import MyTrainer
import pprint

def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True,
            help='Model file name to continue.'
        )

    p.add_argument(
        '--model_name',
        required=not is_continue,
        help='Model file name to save. Additional information would be annotated to the file name.'
    )
    # p.add_argument(
    #     '--train',
    #     required=not is_continue,
    #     help='Training set file name except the extention. (ex: train.en --> train)'
    # )
    # p.add_argument(
    #     '--valid',
    #     required=not is_continue,
    #     help='Validation set file name except the extention. (ex: valid.en --> valid)'
    # )
    p.add_argument(
        '--path',
        required=True
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
    )
    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=10,
        help='Number of epochs to train. Default=%(default)s'
    )
    p.add_argument(
        '--init_epoch',
        required=is_continue,
        type=int,
        default=1,
        help='Set initial epoch number, which can be useful in continue training. Default=%(default)s'
    )
    p.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum length of the training sequence. Default=%(default)s'
    )
    p.add_argument(
        '--lr',
        type=float,
        default=1.,
        help='Initial learning rate. Default=%(default)s',
    )
    p.add_argument(
        '--lr_step',
        type=int,
        default=1,
        help='Number of epochs for each learning rate decay. Default=%(default)s',
    )

    config = p.parse_args()

    return config


def get_model(model, lr):
    model = NewsSummaryModel(model=model, lr=lr)
    return model


# def save_model(pl_trainer):
#     trained_model = NewsSummaryModel.load_from_checkpoint(
#         pl_trainer.checkpoint_callback.best_model_path
#     )
#     trained_model.freeze()


def main(config):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    # data_module = NewsSummaryDataModule(config.train, config.valid, config.model_name, batch_size=config.batch_size, text_max_token_len=config.max_length)
    data_module = NewsSummaryDataModule(config.path, config.model_name, batch_size=config.batch_size, text_max_token_len=config.max_length)
    model = get_model(model=config.model_name)
    trainer = MyTrainer(config)
    trainer.train(model, data_module)
    # save_model(pl_trainer)



if __name__ == '__main__':
    config = define_argparser()
    main(config)