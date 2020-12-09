from clevr_prep.data.get_ground_truth_layout import add_gt_layout
import json
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
    SharpenLossScaler,
    batch_feat_grid2bbox,
    batch_bbox_iou,
)


class Model(pl.LightningModule):
    """
    Neural Module network, implemented as a lightning module for ease of use
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
        # metrics
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        # for loss calc
        self.sharpen_loss_scaler = SharpenLossScaler(self.cfg)

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

    ## bbox offset loss is just MSE

    def sharpen_loss(self, module_probs):
        flat_probs = module_probs.view(-1, self.num_module)
        # the entropy of the module weights
        entropy = -((torch.log(torch.clamp(flat_probs, min=1e-5)) * flat_probs).sum(-1))
        sharpen_loss = torch.mean(entropy)
        return sharpen_loss

    def loc_loss(
        self, loc_scores, bbox_ind_batch, bbox_offset, module_logits, gt_layout
    ):
        sharpen_scale = (
            self.sharpen_loss_scaler(self.global_step) * self.cfg.TRAIN.VQA_LOSS_WEIGHT
        )
        loss = (
            F.cross_entropy(loc_scores, bbox_ind_batch)
            * self.cfg.TRAIN.BBOX_IND_LOSS_WEIGHT
        )
        loss += (
            F.mse_loss(bbox_ind_batch, bbox_offset)
            * self.cfg.TRAIN.BBOX_OFFSET_LOSS_WEIGHT
        )
        if self.cfg.TRAIN.USE_GT_LAYOUT:
            loss += (
                F.cross_entropy(
                    module_logits.view(-1, module_logits.size(2)), gt_layout.view(-1)
                )
                * self.cfg.TRAIN.LAYOUT_LOSS_WEIGHT
            )
        if self.cfg.TRAIN.USE_SHARPEN_LOSS:
            loss += (
                self.sharpen_loss(module_logits)
                * sharpen_scale
                * self.cfg.TRAIN.SHARPEN_LOSS_WEIGHT
            )
        return loss

    def loss(self, answer_logits, answer_idx, module_logits, gt_layout):
        sharpen_scale = (
            self.sharpen_loss_scaler(self.global_step) * self.cfg.TRAIN.VQA_LOSS_WEIGHT
        )
        loss = F.cross_entropy(answer_logits, answer_idx)
        if self.cfg.TRAIN.USE_SHARPEN_LOSS:
            loss += (
                self.sharpen_loss(module_logits)
                * sharpen_scale
                * self.cfg.TRAIN.SHARPEN_LOSS_WEIGHT
            )
        if self.cfg.TRAIN.USE_GT_LAYOUT:
            loss += (
                F.cross_entropy(
                    module_logits.view(-1, module_logits.size(2)), gt_layout.view(-1)
                )
                * self.cfg.TRAIN.LAYOUT_LOSS_WEIGHT
            )
        return loss

    ## below is the pytorch lightning training code

    def training_step(self, batch, batch_idx):
        question_inds = batch["question_inds"]
        seq_length = batch["seq_length"]
        image_feat = batch["image_feat"]
        answer_idx = batch["answer_idx"]
        gt_layout = batch.get("layout_inds", None)
        bbox_ind = batch.get("bbox_ind", None)
        bbox_gt = batch.get("bbox_batch", None)
        question_mask = sequence_mask(seq_length)
        # build loc is the flag telling us whether we are using clevr or clevr-ref
        if not self.cfg.cfg.MODEL.BUILD_LOC:
            output_logits, module_logits = self.forward(
                question_inds, question_mask, image_feat
            )
            loss = self.loss(output_logits, answer_idx, module_logits, gt_layout)
            # logging
            self.train_acc(output_logits, answer_idx)
            self.log("train/loss", loss, on_epoch=True)
            self.log("train/acc", self.train_acc, on_epoch=True)
        else:
            loc_scores, bbox_offset, _, module_logits = self.forward(
                question_inds, question_mask, image_feat
            )
            loss = self.loc_loss(
                loc_scores, bbox_ind, bbox_offset, module_logits, gt_layout
            )
            img_h, img_w, stride_h, stride_w = self.img_sizes
            bbox_pred = batch_feat_grid2bbox(
                torch.argmax(loc_scores, 1),
                bbox_offset,
                stride_h,
                stride_w,
                img_h,
                img_w,
            )
            accuracy = torch.mean(
                batch_bbox_iou(bbox_pred, bbox_gt) >= self.cfg.TRAIN.BBOX_IOU_THRESH
            )
            self.log("train/loss", loss, on_epoch=True)
            self.log("train/acc", accuracy, on_epoch=True)
        return loss

    def _test_step_vqa(self, batch):
        question_inds = batch["question_inds"]
        seq_length = batch["seq_length"]
        image_feat = batch["image_feat"]
        answer_idx = batch.get("answer_idx", None)
        question_mask = sequence_mask(seq_length)
        output_logits, module_logits = self.forward(
            question_inds, question_mask, image_feat
        )
        return output_logits, module_logits, answer_idx

    def _test_step_loc(self, batch, test=False):
        question_inds = batch["question_inds"]
        seq_length = batch["seq_length"]
        image_feat = batch["image_feat"]
        bbox_ind = batch.get("bbox_ind", None)
        bbox_gt = batch.get("bbox_batch", None)
        question_mask = sequence_mask(seq_length)
        loc_scores, bbox_offset, _, module_logits = self.forward(
            question_inds, question_mask, image_feat
        )
        return loc_scores, bbox_offset, module_logits, bbox_ind, bbox_gt

    def validation_step(self, batch, batch_idx):
        gt_layout = batch.get("layout_inds", None)
        # build loc is the flag telling us whether we are using clevr or clevr-ref
        if not self.cfg.MODEL.BUILD_LOC:
            answer_logits, module_logits, answer_idx = self._test_step(batch)
            # logging
            loss = self.loss(answer_logits, answer_idx, module_logits, gt_layout)
            self.valid_acc(answer_logits, answer_idx)
            self.log("valid/loss_epoch", loss, on_step=False, on_epoch=True)
            self.log("valid/acc_epoch", self.valid_acc, on_step=False, on_epoch=True)
            return answer_logits
        else:
            loc_scores, bbox_offset, module_logits, bbox_ind, bbox_gt = self._test_step(
                batch
            )
            loss = self.loc_loss(
                loc_scores, bbox_ind, bbox_offset, module_logits, gt_layout
            )
            img_h, img_w, stride_h, stride_w = self.img_sizes
            bbox_pred = batch_feat_grid2bbox(
                torch.argmax(loc_scores, 1),
                bbox_offset,
                stride_h,
                stride_w,
                img_h,
                img_w,
            )
            accuracy = torch.mean(
                batch_bbox_iou(bbox_pred, bbox_gt) >= self.cfg.TRAIN.BBOX_IOU_THRESH
            )
            self.log("valid/loss", loss, on_epoch=True)
            self.log("valid/acc", accuracy, on_epoch=True)
            return None

    def test_step(self, batch, batch_idx):
        if not self.cfg.MODEL.BUILD_LOC:
            # no answers in test. Instead we just save our predictions
            answer_logits, _, _ = self._test_step(batch, test=True)
            answer_preds = F.softmax(answer_logits, dim=-1)
            return answer_preds

    def validation_epoch_end(self, validation_step_outputs):
        if not self.cfg.MODEL.BUILD_LOC:
            flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
            self.logger.experiment.log(
                {
                    "valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
                    "global_step": self.global_step,
                }
            )

    def test_epoch_end(self, test_step_outputs):
        if not self.cfg.MODEL.BUILD_LOC:
            flattened_preds = (
                torch.flatten(torch.cat(test_step_outputs)).cpu().numpy().tolist()
            )
            with open("test_preds.json", "w") as w:
                json.dump(flattened_preds, w)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.TRAIN.SOLVER.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        return optimizer
