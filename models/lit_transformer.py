from transformer import Transformer, build_transformer, ProjectionLayer
import torch.nn as nn
import lightning as L
import torch


class LT_model(L.LightningModule):

    def __init__(self, config):
        super().__init__()
        pass
