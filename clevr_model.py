import json
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

from model import Model
from utils import sequence_mask, SharpenLossScaler


class ClevrModel(pl.LightningModule):
    """
    Training code for clevr-vqa setup. Uses EMA and a few other nice
    little tricks.
    """

    def __init__(self, cfg, num_choices, module_names, num_vocab):
        super().__init__()
        self.cfg = cfg
        self.num_choices = num_choices
        self.module_names = module_names
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

    # not used here, just a shell around the online model.
    def forward(self, question, question_mask, image_feats):
        return self.online_model(question, question_mask, image_feats)

    def sharpen_loss(self, module_probs):
        flat_probs = module_probs.view(-1, self.num_module)
        # the entropy of the module weights
        entropy = -((torch.log(torch.clamp(flat_probs, min=1e-5)) * flat_probs).sum(-1))
        sharpen_loss = torch.mean(entropy)
        return sharpen_loss

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
        question_mask = sequence_mask(seq_length)
        output_logits, module_logits = self.online_model(
            question_inds, question_mask, image_feat
        )
        loss = self.loss(output_logits, answer_idx, module_logits, gt_layout)
        accumulate(self.offline_model, self.online_model)
        # logging
        self.train_acc(output_logits, answer_idx)
        self.log("train/loss", loss, on_epoch=True)
        self.log("train/acc", self.train_acc, on_epoch=True)
        return loss

    def _test_step(self, batch, test=False):
        question_inds = batch["question_inds"]
        seq_length = batch["seq_length"]
        image_feat = batch["image_feat"]
        if not test:
            answer_idx = batch["answer_idx"]
        else:
            answer_idx = None
        question_mask = sequence_mask(seq_length)
        output_logits, module_logits = self.offline_model(
            question_inds, question_mask, image_feat
        )
        return output_logits, module_logits, answer_idx

    def validation_step(self, batch, batch_idx):
        answer_logits, module_logits, answer_idx = self._test_step(batch)
        gt_layout = batch.get("layout_inds", None)
        loss = self.loss(answer_logits, answer_idx, module_logits, gt_layout)
        self.valid_acc(answer_logits, answer_idx)
        self.log("valid/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("valid/acc_epoch", self.valid_acc, on_step=False, on_epoch=True)
        return answer_logits

    def test_step(self, batch, batch_idx):
        # no answers in test. Instead we just save our predictions
        answer_logits, _, _ = self._test_step(batch, test=True)
        answer_preds = F.softmax(answer_logits, dim=-1)
        return answer_preds

    def validation_epoch_end(self, validation_step_outputs):
        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {
                "valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "global_step": self.global_step,
            }
        )

    def test_epoch_end(self, test_step_outputs):
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


# ema accumulation
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=(1 - decay))
