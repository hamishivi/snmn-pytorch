import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.Functional as F
import pytorch_lightning as pl

from .config import cfg
from controller import Controller
from nmn import NMN
from . import controller, nmn, input_unit, output_unit, vis
from utils import pack_and_rnn, get_positional_encoding

class Model(pl.LightningModule):
    """
    Neural Module network, implemented as a lightning module for ease of use
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_choices = cfg.NUM_CHOICES
        self.steps = cfg.STEPS
        self.q_enc = nn.GRU(cfg.QENC.DIM, cfg.SNMN.DIM, bidirectional=True, batch_first=True)
        self.module_names = ["find"]
        self.num_module = len(self.module_names)
        self.controller = Controller(cfg)
        self.pe_dim = cfg.pe_dim
        self.kb_process = Sequential(
            nn.Conv2d(cfg.KB.DIM*2, cfg.SNMN.DIM, 1, 1),
            nn.ELU(),
            nn.Conv2d(cfg.SNMN.DIM, cfg.SNMN.DIM, 1, 1)
        )
        self.nmn = NMN(self.module_names)
        self.output_unit = Sequential(
            nn.Linear(cfg.SNMN.DIM*2, cfg.OUTPUT.DIM),
            nn.ELU(),
            nn.Linear(cfg.OUTPUT.DIM, self.num_choices)
        )

        self.init_ctrl = nn.Parameter(torch.ones(cfg.SNMN.DIM))

    def forward(self, question, question_mask, image_feats):
        # Input unit - encoding question
        # todo: question embedding
        lstm_seq, (h, _) = pack_and_rnn(question, question_mask.sum(1), self.q_enc)
        q_vec = torch.cat([h[0], h[1]], -1)
        # Process kb
        position_encoding = get_positional_encoding(image_feats.size(1), image_feats.size(2), self.pe_dim)
        # convert to tensor and tile along batch dim
        position_encoding = torch.tensor(position_encoding, device=image_feats.device).repeat(image_feats.size(0), 1, 1, 1)
        kb_batch = self.kb_process(torch.cat([image_feats, position_encoding], 3))
        # init values
        control = self.init_ctrl.unsqueeze(0).repeat(image_feats.size(0), 1)
        mem, att_stack, stack_ptr = self.nmn.get_init_values()
        for i in range(self.steps):
            # Controller and NMN
            control, module_probs = self.controller(lstm_seq, q_vec, control, question_mask, i)
            mem, att_stack, stack_ptr = self.nmn(control, kb_batch, module_probs, mem, att_stack, stack_ptr)
        # output - two layer FC
        output_logits = self.output_unit(torch.cat([q_vec, mem], 1))
        return output_logits

    def training_step(self, batch, batch_idx):
        output_logits = self.forward(batch[0], batch[1], batch[2])
        loss = F.cross_entropy(output_logits, batch[3])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    ## bbox offset loss is just MSE

    def sharpen_loss(self, module_probs):
        flat_probs = module_probs.view(-1, self.num_module)
        # the entropy of the module weights
        entropy = -((torch.log(torch.maximum(flat_probs, 1e-5)) * flat_probs).sum(-1))
        sharpen_loss = torch.mean(entropy)
        return sharpen_loss
