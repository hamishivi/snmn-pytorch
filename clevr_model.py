import json
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model import Model
from utils import sequence_mask, SharpenLossScaler, batch_feat_grid2bbox, batch_bbox_iou


class ClevrModel(pl.LightningModule):
    """
    Training code for clevr-vqa setup. Uses EMA and a few other nice
    little tricks.
    """

    def __init__(self, cfg, num_choices, module_names, num_vocab, img_sizes):
        super().__init__()
        self.cfg = cfg
        self.num_choices = num_choices
        self.module_names = module_names
        self.img_sizes = img_sizes
        self.steps = cfg.MODEL.T_CTRL
        self.num_module = len(module_names)
        # models (ema and reg)
        self.online_model = Model(cfg, num_choices, module_names, num_vocab)
        self.offline_model = Model(cfg, num_choices, module_names, num_vocab)
        accumulate(self.offline_model, self.online_model, 0)
        # metrics
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        # for loss calc
        self.sharpen_loss_scaler = SharpenLossScaler(self.cfg)

    # used for basic inference
    def forward(self, question, question_mask, image_feats):
        return self.online_model(question, question_mask, image_feats)

    def sharpen_loss(self, module_logits):
        module_probs = F.softmax(module_logits, dim=-1)
        sharpen_scale = self.sharpen_loss_scaler(self.global_step)
        flat_probs = module_probs.view(-1, self.num_module)
        # the entropy of the module weights
        entropy = -((torch.log(torch.clamp(flat_probs, min=1e-5)) * flat_probs).sum(-1))
        sharpen_loss = torch.mean(entropy)
        return sharpen_loss * sharpen_scale * self.cfg.TRAIN.SHARPEN_LOSS_WEIGHT

    def loc_loss(self, loc_scores, bbox_offset_fcn, bbox_ind_batch, bbox_offset):
        loss = (
            F.cross_entropy(loc_scores, bbox_ind_batch)
            * self.cfg.TRAIN.BBOX_IND_LOSS_WEIGHT
        )
        N = bbox_offset_fcn.size(0)
        B = bbox_offset_fcn.size(1)
        bbox_offset_flat = bbox_offset_fcn.view(N * B, 4)
        slice_inds = (
            torch.arange(0, N, device=self.device) * B + bbox_ind_batch
        ).long()
        bbox_offset_sliced = bbox_offset_flat[slice_inds]
        loss += (
            F.mse_loss(bbox_offset_sliced.float(), bbox_offset.float())
            * self.cfg.TRAIN.BBOX_OFFSET_LOSS_WEIGHT
        )
        return loss

    def vqa_loss(self, answer_logits, answer_idx):
        return (
            F.cross_entropy(answer_logits, answer_idx) * self.cfg.TRAIN.VQA_LOSS_WEIGHT
        )

    def gt_loss(self, module_logits, gt_layout):
        return (
            F.cross_entropy(
                module_logits.reshape(-1, module_logits.size(2)), gt_layout.reshape(-1)
            )
            * self.cfg.TRAIN.LAYOUT_LOSS_WEIGHT
        )

    ## below is the pytorch lightning training code
    def training_step(self, batch, batch_idx, use_sharpen=True):
        question_inds = batch["question_inds"]
        seq_length = batch["seq_length"]
        image_feat = batch["image_feat"]
        answer_idx = batch.get("answer_idx", None)
        gt_layout = batch.get("layout_inds", None)
        bbox_ind = batch.get("bbox_ind", None)
        bbox_gt = batch.get("bbox_batch", None)
        bbox_offset = batch.get("bbox_offset", None)
        question_mask = sequence_mask(seq_length)
        outputs = self.online_model(question_inds, question_mask, image_feat)
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float)
        # we support training on vqa only, loc only, or both, depending on these flags.
        if self.cfg.MODEL.BUILD_VQA and answer_idx is not None:
            loss += self.vqa_loss(outputs["logits"], answer_idx)
            self.train_acc(outputs["logits"], answer_idx)
            self.log("train/vqa_acc", self.train_acc)
        if self.cfg.MODEL.BUILD_LOC and bbox_ind is not None:
            loss += self.loc_loss(
                outputs["loc_scores"], outputs["bbox_offset_fcn"], bbox_ind, bbox_offset
            )
            feat_h, feat_w, _, _, stride_h, stride_w = self.img_sizes
            bbox_pred = batch_feat_grid2bbox(
                torch.argmax(outputs["loc_scores"], 1),
                outputs["bbox_offset"],
                stride_h,
                stride_w,
                feat_h,
                feat_w,
            )
            accuracy = torch.mean(
                (
                    batch_bbox_iou(bbox_pred, bbox_gt) >= self.cfg.TRAIN.BBOX_IOU_THRESH
                ).float()
            )
            self.log("train/loc_acc", accuracy)
        if self.cfg.TRAIN.USE_SHARPEN_LOSS and use_sharpen:
            loss += self.sharpen_loss(outputs["module_logits"])
        if self.cfg.TRAIN.USE_GT_LAYOUT:
            loss += self.gt_loss(outputs["module_logits"], gt_layout)
        self.log("train/loss", loss)
        # technically this means the offline model is behind, but its fine.
        accumulate(self.offline_model, self.online_model)
        return loss

    def _test_step(self, batch):
        question_inds = batch["question_inds"]
        seq_length = batch["seq_length"]
        image_feat = batch["image_feat"]
        question_mask = sequence_mask(seq_length)
        outputs = self.offline_model(question_inds, question_mask, image_feat)
        return outputs

    def validation_step(self, batch, batch_idx, use_sharpen=True):
        gt_layout = batch.get("layout_inds", None)
        answer_idx = batch.get("answer_idx", None)
        bbox_ind = batch.get("bbox_ind", None)
        bbox_gt = batch.get("bbox_batch", None)
        bbox_offset = batch.get("bbox_offset", None)
        outputs = self._test_step(batch)
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float)
        # we support training on vqa only, loc only, or both, depending on these flags.
        if self.cfg.MODEL.BUILD_VQA and answer_idx is not None:
            loss += self.vqa_loss(outputs["logits"], answer_idx)
            self.valid_acc(outputs["logits"], answer_idx)
            self.log("valid/vqa_acc", self.valid_acc, on_step=False, on_epoch=True)
        if self.cfg.MODEL.BUILD_LOC and bbox_ind is not None:
            loss += self.loc_loss(
                outputs["loc_scores"], outputs["bbox_offset_fcn"], bbox_ind, bbox_offset
            )
            feat_h, feat_w, _, _, stride_h, stride_w = self.img_sizes
            bbox_pred = batch_feat_grid2bbox(
                torch.argmax(outputs["loc_scores"], 1),
                outputs["bbox_offset"],
                stride_h,
                stride_w,
                feat_h,
                feat_w,
            )
            accuracy = torch.mean(
                (
                    batch_bbox_iou(bbox_pred, bbox_gt) >= self.cfg.TRAIN.BBOX_IOU_THRESH
                ).float()
            )
            self.log("valid/loc_acc", accuracy, on_step=False, on_epoch=True)
        if self.cfg.TRAIN.USE_SHARPEN_LOSS and use_sharpen:
            loss += self.sharpen_loss(outputs["module_logits"])
        if self.cfg.TRAIN.USE_GT_LAYOUT:
            loss += self.gt_loss(outputs["module_logits"], gt_layout)
        self.log("valid/loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_epoch_end(self, test_step_outputs):
        if self.cfg.MODEL.BUILD_VQA:
            flattened_preds = (
                torch.flatten(torch.cat(test_step_outputs)).cpu().numpy().tolist()
            )
            with open("test_preds.json", "w") as w:
                json.dump(flattened_preds, w)

    def configure_optimizers(self):
        # the og code does no-bias regularisation!
        biases = [param for name, param in self.online_model.named_parameters() if 'bias' in name]
        weights = [param for name, param in self.online_model.named_parameters() if 'bias' not in name]
        optimizer = torch.optim.Adam(
            [
                { 'params': biases, 'weight_decay': 0 },
                { 'params': weights, 'weight_decay': self.cfg.TRAIN.WEIGHT_DECAY }

            ],
            lr=self.cfg.TRAIN.SOLVER.LR,
        )
        return optimizer

# to override pytorch default inits
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ema accumulation
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=(1 - decay))
