import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F
import pytorch_lightning as pl

from controller import Controller
from nmn import NMN
from utils import pack_and_rnn, get_positional_encoding, sequence_mask


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
        self.controller = Controller(cfg)
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

        self.init_ctrl = nn.Parameter(torch.ones(cfg.MODEL.LSTM_DIM))

    def forward(self, question, question_mask, image_feats):
        # Input unit - encoding question
        question = self.q_embed(question)
        lstm_seq, (h, _) = pack_and_rnn(question, question_mask.sum(1), self.q_enc)
        q_vec = torch.cat([h[0], h[1]], -1)
        # Process kb
        position_encoding = get_positional_encoding(
            image_feats.size(1), image_feats.size(2), self.pe_dim
        )
        # convert to tensor and tile along batch dim
        position_encoding = torch.tensor(
            position_encoding, device=image_feats.device
        ).repeat(image_feats.size(0), 1, 1, 1)
        kb_batch = (
            torch.cat([image_feats, position_encoding], 3).permute([0, 3, 1, 2]).float()
        )
        kb_batch = self.kb_process(kb_batch)
        # init values
        control = self.init_ctrl.unsqueeze(0).repeat(image_feats.size(0), 1)
        mem, att_stack, stack_ptr = self.nmn.get_init_values()
        module_logits = []
        for i in range(self.steps):
            # Controller and NMN
            control, module_probs = self.controller(
                lstm_seq, q_vec, control, question_mask, i
            )
            module_logits.append(module_probs)
            mem, att_stack, stack_ptr = self.nmn(
                control, kb_batch, module_probs, mem, att_stack, stack_ptr
            )
        # output - two layer FC
        output_logits = self.output_unit(torch.cat([q_vec, mem], 1))
        return output_logits, module_logits

    def training_step(self, batch, batch_idx):
        question_inds, seq_length, image_feat, image_path, answer_idx, layout_inds = (
            batch
        )
        question_mask = sequence_mask(seq_length)
        output_logits, module_logits = self.forward(
            question_inds, question_mask, image_feat
        )
        loss = F.cross_entropy(output_logits, answer_idx)
        loss += self.sharpen_loss(module_logits)
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
