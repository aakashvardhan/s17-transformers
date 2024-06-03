import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from train import get_all_sentenses, get_or_build_tokenizer, get_ds


class BillingualDataset(Dataset):
    """
    A PyTorch dataset for handling bilingual text data.

    Args:
        ds (Dataset): The original dataset containing the bilingual text data.
        tokenizer_src (Tokenizer): The tokenizer for the source language.
        tokenizer_tgt (Tokenizer): The tokenizer for the target language.
        src_lang (str): The source language.
        tgt_lang (str): The target language.
        seq_len (int): The maximum sequence length.

    Attributes:
        seq_len (int): The maximum sequence length.
        ds (Dataset): The original dataset containing the bilingual text data.
        tokenizer_src (Tokenizer): The tokenizer for the source language.
        tokenizer_tgt (Tokenizer): The tokenizer for the target language.
        src_lang (str): The source language.
        tgt_lang (str): The target language.
        sos_token (torch.Tensor): The tensor representing the start-of-sequence token.
        eos_token (torch.Tensor): The tensor representing the end-of-sequence token.
        pad_token (torch.Tensor): The tensor representing the padding token.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.

    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the encoder input, decoder input, encoder mask, decoder mask, label,
                  source text, and target text.

        Raises:
            ValueError: If the sentence is too long.

        """
        src_tgt_pair = self.ds[idx]
        src_text = src_tgt_pair["translation"][self.src_lang]
        tgt_text = src_tgt_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence too long")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            & casual_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def casual_mask(size):
    """
    Generate a causal mask for self-attention mechanism.

    Args:
        size (int): The size of the mask.

    Returns:
        torch.Tensor: The causal mask with shape (1, size, size), where the upper triangle values are set to 0 and the lower triangle values are set to 1.

    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class LT_DataModule(L.LightningDataModule):
    """
    LightningDataModule for handling data loading and processing in the LT_DataModule class.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        (
            self.train_dataloader,
            self.val_dataloader,
            self.tokenizer_src,
            self.tokenizer_tgt,
        ) = get_ds(self.config)

        self.config["src_vocab_size"] = self.tokenizer_src.get_vocab_size()
        self.config["tgt_vocab_size"] = self.tokenizer_tgt.get_vocab_size()

    def train_dataloader(self):
        """
        This function is likely intended to create a data loader for training a machine learning model.
        """
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader

    def get_tokenizers(self):
        """
        Returns the source and target tokenizers used in the dataset.

        Returns:
            tokenizer_src (object): The tokenizer used for the source language.
            tokenizer_tgt (object): The tokenizer used for the target language.
        """
        return self.tokenizer_src, self.tokenizer_tgt
