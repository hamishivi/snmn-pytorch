import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.modules import module
import numpy as np

from controller import Controller
from nmn import NMN
from utils import pack_and_rnn, get_positional_encoding, channels_last_conv


class Model(pl.LightningModule):
    """
    Neural Module network, implemented as a lightning module for ease of use.
    """

    def __init__(self, cfg, num_choices, module_names, num_vocab, img_sizes):
        super().__init__()
        self.cfg = cfg
        self.num_choices = num_choices
        self.module_names = module_names
        self.img_sizes = img_sizes
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
        # basic vqa output for clevr
        self.output_unit = Sequential(
            nn.Linear(
                cfg.MODEL.NMN.MEM_DIM + cfg.MODEL.LSTM_DIM, cfg.MODEL.VQA_OUTPUT_DIM
            ),
            nn.ELU(),
            nn.Linear(cfg.MODEL.VQA_OUTPUT_DIM, self.num_choices),
        )
        # output for clevr-ref
        self.output_loc_aff_w = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.ones(cfg.MODEL.H_FEAT, cfg.MODEL.W_FEAT, 1)
            )
        )
        self.output_loc_aff_b = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.ones(cfg.MODEL.H_FEAT, cfg.MODEL.W_FEAT, 1)
            )
        )
        self.loc_conv = nn.Conv2d(cfg.MODEL.KB_DIM, 4, 1, 1)

        control_dim = cfg.MODEL.KB_DIM
        if cfg.MODEL.CTRL.USE_WORD_EMBED:
            control_dim = cfg.MODEL.EMBED_DIM
        # random normal with std dev 1/sqrt(ctrl dim)
        self.init_ctrl = nn.Parameter(
            torch.randn(control_dim) * np.sqrt(1 / cfg.MODEL.LSTM_DIM)
        )

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
        # output for clevr-ref
        if self.cfg.MODEL.BUILD_LOC:
            att_last = self.nmn.get_stack_value(att_stack, stack_ptr)
            # first a linear layer (LOC_SCORES_POS_AFFINE)
            loc_scores = self.output_loc_aff_w * att_last + self.output_loc_aff_b
            # one layer conv (BBOX_REG_AS_FCN)
            bbox_offset_fcn = channels_last_conv(kb_batch, self.loc_conv)
            N = bbox_offset_fcn.size(0)
            B = self.cfg.MODEL.H_FEAT * self.cfg.MODEL.W_FEAT
            # bbox_offset_fcn [N, B, 4] is used for training
            bbox_offset_fcn = bbox_offset_fcn.view(N, B, 4)
            # bbox_offset [N, 4] is only used for prediction
            bbox_offset_flat = bbox_offset_fcn.view(N * B, 4)
            slice_inds = torch.range(0, N) * B + torch.argmax(loc_scores, dim=-1).long()
            bbox_offset = torch.gather(bbox_offset_flat, slice_inds)
            return loc_scores, bbox_offset, bbox_offset_fcn, torch.stack(module_logits)
        return output_logits, torch.stack(module_logits)
