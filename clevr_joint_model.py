import json
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

from model import Model
from clevr_model import ClevrModel


class ClevrJointModel(ClevrModel):
    """
    Model for joint training, since this uses a different setup to the loc and vqa datasets.
    Inherits from ClevrModel, so only training code is changed to split up the two batches.
    """

    def __init__(self, cfg, num_choices, module_names, num_vocab, img_sizes):
        super().__init__(cfg, num_choices, module_names, num_vocab, img_sizes)

    def training_step(self, batch, batch_idx):
        vqa_batch = batch["vqa"]
        loss = super().training_step(vqa_batch, batch_idx)
        loc_batch = batch["loc"]
        loss += super().training_step(loc_batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        vqa_batch = batch["vqa"]
        loss = super().training_step(vqa_batch, batch_idx)
        loc_batch = batch["loc"]
        loss += super().training_step(loc_batch, batch_idx)
        return loss
