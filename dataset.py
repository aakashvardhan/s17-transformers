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
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype = torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_tgt_pair = self.ds[idx]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        #For encoding, we PAD both SOS and EOS. For decoding, we only pad SOS.
        #THe model is required to predict EOS and stop on its own.
        
        #Make sure that padding is not negative (ie the sentance is too long)
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence too long")
            
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype = torch.int64)
            ],
            dim =  0,
        )
        
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0,
        )
        
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype = torch.int64),
            ],
            dim = 0,
        )
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), 
            # encoder mask: (1, 1, seq_len) -> Has 1 when there is text and 0 when there is pad (no text)
            
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            # (1, seq_len) and (1, seq_len, seq_len)
            # Will get 0 for all pads. And 0 for earlier text.
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
            }
    
def casual_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal = 1).type(torch.int)
    #This will get the upper traingle values
    return mask == 0


class LT_DataModule(L.LightningDataModule):
    """
    LightningDataModule for handling data loading and processing in the LT_DataModule class.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def setup(self, stage=None):
        self.train_dataloader, self.val_dataloader, self.tokenizer_src, self.tokenizer_tgt = get_ds(self.config)
        
        self.config['src_vocab_size'] = self.tokenizer_src.get_vocab_size()
        self.config['tgt_vocab_size'] = self.tokenizer_tgt.get_vocab_size()
        
    def train_dataloader(self):
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