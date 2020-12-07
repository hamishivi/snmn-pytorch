import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.modules import module
import numpy as np
import wandb

from controller import Controller
from nmn import NMN
from utils import (
    pack_and_rnn,
    get_positional_encoding,
    sequence_mask,
    channels_last_conv,
)


class Model(pl.LightningModule):
    """
    Neural Module network, implemented as a lightning module for ease of use
    """

    def __init__(self, cfg, num_choices, module_names, num_vocab):
        super().__init__()
        self.cfg = cfg
        self.num_choices = num_choices
        self.module_names = module_names
        self.steps = cfg.MODEL.T_CTRL
        self.q_embed = nn.Embedding(num_vocab, cfg.MODEL.EMBED_DIM)
        self.q_enc = nn.LSTM(
            cfg.MODEL.EMBED_DIM,
            cfg.MODEL.LSTM_DIM // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.num_module = len(module_names)
        self.controller = Controller(cfg, self.num_module)
        self.pe_dim = cfg.MODEL.PE_DIM
        self.kb_process = Sequential(
            nn.Conv2d(cfg.MODEL.KB_DIM * 2 + cfg.MODEL.PE_DIM, cfg.MODEL.KB_DIM, 1, 1),
            nn.ELU(),
            nn.Conv2d(cfg.MODEL.KB_DIM, cfg.MODEL.KB_DIM, 1, 1),
        )
        self.nmn = NMN(cfg, self.module_names)
        self.output_unit = Sequential(
            nn.Linear(
                cfg.MODEL.NMN.MEM_DIM + cfg.MODEL.LSTM_DIM, cfg.MODEL.VQA_OUTPUT_DIM
            ),
            nn.ELU(),
            nn.Linear(cfg.MODEL.VQA_OUTPUT_DIM, self.num_choices),
        )

        # random normal with std dev 1/sqrt(ctrl dim)
        self.init_ctrl = nn.Parameter(
            torch.randn(cfg.MODEL.LSTM_DIM) * np.sqrt(1 / cfg.MODEL.LSTM_DIM)
        )
        # metrics
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, question, question_mask, image_feats):
        # Input unit - encoding question
        question = self.q_embed(question)
        lstm_seq, (h, _) = pack_and_rnn(question, question_mask.sum(1), self.q_enc)
        q_vec = torch.cat([h[0], h[1]], -1)
        # Process kb
        position_encoding = get_positional_encoding(
            image_feats.size(1), image_feats.size(2), self.pe_dim, image_feats.device
        ).repeat(image_feats.size(0), 1, 1, 1)
        kb_batch = torch.cat([image_feats, position_encoding], 3)
        kb_batch = channels_last_conv(kb_batch, self.kb_process)
        # init values
        control = self.init_ctrl.unsqueeze(0).repeat(image_feats.size(0), 1)
        att_stack, stack_ptr, mem = self.nmn.get_init_values(
            image_feats.size(0), image_feats.device
        )
        module_logits = []
        for i in range(self.steps):
            # Controller and NMN
            control, module_probs = self.controller(
                question, lstm_seq, q_vec, control, question_mask, i
            )
            module_logits.append(module_probs)
            att_stack, stack_ptr, mem = self.nmn(
                control, kb_batch, module_probs, mem, att_stack, stack_ptr
            )
        # output - two layer FC
        output_logits = self.output_unit(torch.cat([q_vec, mem], 1))
        return output_logits, torch.stack(module_logits)

    ## bbox offset loss is just MSE

    def sharpen_loss(self, module_probs):
        flat_probs = module_probs.view(-1, self.num_module)
        # the entropy of the module weights
        entropy = -((torch.log(torch.clamp(flat_probs, min=1e-5)) * flat_probs).sum(-1))
        sharpen_loss = torch.mean(entropy)
        return sharpen_loss

    def loss(self, answer_logits, answer_idx, module_logits):
        loss = F.cross_entropy(answer_logits, answer_idx)
        loss += self.sharpen_loss(module_logits)
        return loss

    ## below is the pytorch lightning training code

    def training_step(self, batch, batch_idx):
        question_inds = batch["question_inds"]
        seq_length = batch["seq_length"]
        image_feat = batch["image_feat"]
        answer_idx = batch["answer_idx"]
        question_mask = sequence_mask(seq_length)
        output_logits, module_logits = self.forward(
            question_inds, question_mask, image_feat
        )
        loss = self.loss(output_logits, answer_idx, module_logits)
        # logging
        self.train_acc(output_logits, answer_idx)
        self.log("train/loss", loss, on_epoch=True)
        self.log("train/acc", self.train_acc, on_epoch=True)
        return loss

    def _test_step(self, batch):
        question_inds = batch["question_inds"]
        seq_length = batch["seq_length"]
        image_feat = batch["image_feat"]
        answer_idx = batch["answer_idx"]
        question_mask = sequence_mask(seq_length)
        output_logits, module_logits = self.forward(
            question_inds, question_mask, image_feat
        )
        return output_logits, module_logits, answer_idx

    def validation_step(self, batch, batch_idx):
        answer_logits, module_logits, answer_idx = self._test_step(batch)
        loss = self.loss(answer_logits, answer_idx, module_logits)
        self.valid_acc(answer_logits, answer_idx)
        self.log("valid/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("valid/acc_epoch", self.valid_acc, on_step=False, on_epoch=True)
        return answer_logits

    def test_step(self, batch, batch_idx):
        answer_logits, module_logits, answer_idx = self._test_step(batch)
        loss = self.loss(answer_logits, answer_idx, module_logits)
        self.test_acc(answer_logits, answer_idx)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

    def validation_epoch_end(self, validation_step_outputs):
        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {
                "valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "global_step": self.global_step,
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.TRAIN.SOLVER.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        return optimizer
