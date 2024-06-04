import os
import warnings
from pathlib import Path
import torch
import torchmetrics
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    TQDMProgressBar,
)


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config_file import get_config, get_weights_file_path
from dataset import BillingualDataset, LT_DataModule, casual_mask

import torch.nn as nn
from models.lit_transformer import LT_model
from models.transformer import build_transformer


torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:12240"
config = get_config()


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len):

    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source_mask).fill_(next_word.item()),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(
        src_vocab_size,
        tgt_vocab_size,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    )
    return model


def main(cfg, ckpt_file, if_ckpt=False):
    """
    Main function for training and evaluating a model.

    Args:
        cfg (dict): Configuration parameters for the model training.
        ckpt_file (str): Path to the checkpoint file.
        if_ckpt (bool, optional): Whether to load the model from a checkpoint. Defaults to False.
    """

    L.seed_everything(42, workers=True)
    print("Seed set to 42...")

    # Initialize the data module
    datamodule = LT_DataModule(cfg)
    datamodule.setup()
    tok_src, tok_tgt = datamodule.get_tokenizers()
    print("DataModule initialized...")

    # Initialize the model
    model = LT_model(cfg, tok_src, tok_tgt)

    # Tensorboard logger
    tb_logger = TensorBoardLogger(
        save_dir=os.getcwd(), version=1, name="lightning_logs"
    )

    # Initialize the trainer

    trainer = L.Trainer(
        precision=cfg["precision"],
        max_epochs=cfg["num_epochs"],
        logger=tb_logger,
        accelerator=cfg["accelerator"],
        devices="auto",
        default_root_dir=cfg["model_folder"],
        callbacks=[
            ModelCheckpoint(
                dirpath=cfg["model_folder"],
                save_top_k=3,
                monitor="train_loss",
                mode="min",
                filename="model-{epoch:02d}-{train_loss:4f}",
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step", log_momentum=True),
            EarlyStopping(monitor="train_loss", mode="min", stopping_threshold=0.15),
            TQDMProgressBar(refresh_rate=10),
        ],
        gradient_clip_val=0.5,
        num_sanity_val_steps=5,
        sync_batchnorm=True,
        enable_progress_bar=True,
        log_every_n_steps=5,
        check_val_every_n_epoch=2,
        limit_val_batches=1000,
    )

    if if_ckpt:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_file)
    else:
        trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model=model, datamodule=datamodule)
    print("Model Evaluation Done...")

    # Save the model
    torch.save(
        model.state_dict(),
        "saved_resnet18_model.pth",
    )
    print("Model saved...")
