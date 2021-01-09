"""
DataLoader class for CLEVR used in training.
Also define DataModule for pytorch-lightning-style training.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from utils import VocabDict
import numpy as np


class PreprocessedClevr(Dataset):
    def __init__(
        self,
        imdb_file,
        vocab_question_file,
        t_encoder,
        t_decoder,
        load_gt_layout,
        vocab_answer_file,
        vocab_layout_file,
        img_h,
        img_w,
        prune_filter_module=True,
    ):
        self.imdb = np.load(imdb_file, allow_pickle=True)
        self.vocab_dict = VocabDict(vocab_question_file)
        self.t_encoder = t_encoder
        # peek one example to see whether answer and gt_layout are in the data
        self.load_answer = (
            "answer" in self.imdb[0] and self.imdb[0]["answer"] is not None
        )
        self.load_bbox = "bbox" in self.imdb[0] and self.imdb[0]["bbox"] is not None
        self.load_gt_layout = load_gt_layout and (
            "gt_layout_tokens" in self.imdb[0]
            and self.imdb[0]["gt_layout_tokens"] is not None
        )
        # load answer dict
        self.answer_dict = VocabDict(vocab_answer_file)
        self.t_decoder = t_decoder
        self.layout_dict = VocabDict(vocab_layout_file)
        if self.load_gt_layout:
            self.prune_filter_module = prune_filter_module
        # load one feature map to peek its size
        feats = np.load(self.imdb[0]["feature_path"])
        self.feat_H, self.feat_W, self.feat_D = feats.shape[1:]
        if self.load_bbox:
            self.img_H = img_h
            self.img_W = img_w
            self.stride_H = self.img_H * 1.0 / self.feat_H
            self.stride_W = self.img_W * 1.0 / self.feat_W

    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):
        iminfo = self.imdb[idx]
        question_inds = [self.vocab_dict.word2idx(w) for w in iminfo["question_tokens"]]
        seq_length = len(question_inds)
        image_feat = np.load(iminfo["feature_path"])
        image_path = iminfo["image_path"]
        if self.load_answer:
            answer_idx = self.answer_dict.word2idx(iminfo["answer"])
        if self.load_bbox:
            bbox_batch = iminfo["bbox"]
            bbox_ind, bbox_offset = bbox2feat_grid(
                iminfo["bbox"], self.stride_H, self.stride_W, self.feat_H, self.feat_W
            )
        if self.load_gt_layout:
            gt_layout_tokens = iminfo["gt_layout_tokens"]
            if self.prune_filter_module:
                # remove duplicated consequtive modules
                # (only keeping one _Filter)
                for n_t in range(len(gt_layout_tokens) - 1, 0, -1):
                    if (
                        gt_layout_tokens[n_t - 1] in {"_Filter", "_Find"}
                        and gt_layout_tokens[n_t] == "_Filter"
                    ):
                        gt_layout_tokens[n_t] = None
                gt_layout_tokens = [t for t in gt_layout_tokens if t]
            layout_inds = [self.layout_dict.word2idx(w) for w in gt_layout_tokens]

        batch = {
            "question_inds": torch.tensor(question_inds),
            "seq_length": torch.tensor(seq_length),
            "image_feat": torch.tensor(image_feat).squeeze(0),
            "image_path": image_path,
        }
        if self.load_answer:
            batch["answer_idx"] = torch.tensor(answer_idx)
        if self.load_bbox:
            batch["bbox_batch"] = torch.tensor(bbox_batch)
            batch["bbox_ind"] = torch.tensor(bbox_ind)
            batch["bbox_offset"] = torch.tensor(bbox_offset)
        if self.load_gt_layout:
            batch["layout_inds"] = torch.tensor(layout_inds)
        return batch

    def get_answer_choices(self):
        return self.answer_dict.num_vocab

    def get_module_names(self):
        return self.layout_dict.word_list

    def get_vocab_size(self):
        return self.vocab_dict.num_vocab

    def get_img_sizes(self):
        return (
            self.feat_H,
            self.feat_W,
            self.img_H,
            self.img_W,
            self.stride_H,
            self.stride_W,
        )


# pytorch doesnt pad in collate, so we use a custom collate fn
def clevr_collate(batch, max_ops, noop_idx):
    batch_dict = {k: [batch[0][k]] for k in batch[0]}
    if len(batch) > 1:
        for k in batch_dict:
            for b in batch[1:]:
                batch_dict[k].append(b[k])
    # stack/padding stuff
    batch_dict["question_inds"] = pad_sequence(
        batch_dict["question_inds"], batch_first=True
    )
    # we pad to the max length with the no-op index.
    if "layout_inds" in batch_dict:
        layout_inds = torch.ones(max_ops, len(batch)) * noop_idx
        for i, b in enumerate(batch):
            layout_inds[:b["layout_inds"].size(0), i] = b["layout_inds"]
        batch_dict["layout_inds"] = layout_inds.long().transpose(0, 1)
    for k in [
        "seq_length",
        "image_feat",
        "answer_idx",
        "bbox_batch",
        "bbox_ind",
        "bbox_offset",
    ]:
        if k in batch_dict:
            batch_dict[k] = torch.stack(batch_dict[k], 0)
    return batch_dict


# dataset code for joint
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, clevr_dataset, clevr_ref_dataset):
        self.clevr = clevr_dataset
        self.clevr_ref = clevr_ref_dataset
        self.layout_dict = clevr_dataset.layout_dict

    def __getitem__(self, i):
        return {"vqa": self.clevr[i], "loc": self.clevr_ref[i]}

    def __len__(self):
        return min(len(self.clevr), len(self.clevr_ref))

    # wrapper for the attributes of the other two
    def get_answer_choices(self):
        return self.clevr.answer_dict.num_vocab

    def get_module_names(self):
        return self.clevr.layout_dict.word_list

    def get_vocab_size(self):
        return self.clevr.vocab_dict.num_vocab

    def get_img_sizes(self):
        return (
            self.clevr_ref.feat_H,
            self.clevr_ref.feat_W,
            self.clevr_ref.img_H,
            self.clevr_ref.img_W,
            self.clevr_ref.stride_H,
            self.clevr_ref.stride_W,
        )


def joint_collate(batch, max_ops, noop_idx):
    vqa_dict = clevr_collate([b["vqa"] for b in batch], max_ops, noop_idx)
    loc_dict = clevr_collate([b["loc"] for b in batch], max_ops, noop_idx)
    return {"vqa": vqa_dict, "loc": loc_dict}


def bbox2feat_grid(bbox, stride_H, stride_W, feat_H, feat_W):
    x1, y1, w, h = bbox
    x2 = x1 + w - 1
    y2 = y1 + h - 1
    # map the bbox coordinates to feature grid
    x1 = x1 * 1.0 / stride_W - 0.5
    y1 = y1 * 1.0 / stride_H - 0.5
    x2 = x2 * 1.0 / stride_W - 0.5
    y2 = y2 * 1.0 / stride_H - 0.5
    xc = np.minimum(np.maximum(np.int32(np.round((x1 + x2) / 2.0)), 0), feat_W - 1)
    yc = np.minimum(np.maximum(np.int32(np.round((y1 + y2) / 2.0)), 0), feat_H - 1)
    ind = yc * feat_W + xc
    offset = x1 - xc, y1 - yc, x2 - xc, y2 - yc
    return ind, offset


class ClevrDataModule(pl.LightningDataModule):
    def __init__(self, cfg, batch_size=64):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size

    def _construct_clevr_(self, filename):
        return PreprocessedClevr(
            filename,
            self.cfg.VOCAB_QUESTION_FILE,
            self.cfg.MODEL.T_ENCODER,
            self.cfg.MODEL.T_CTRL,
            True,
            self.cfg.VOCAB_ANSWER_FILE,
            self.cfg.VOCAB_LAYOUT_FILE,
            self.cfg.MODEL.H_IMG,
            self.cfg.MODEL.W_IMG,
        )

    def _construct_joint_(self, filename_vqa, filename_loc):
        clevr = PreprocessedClevr(
            filename_vqa,
            self.cfg.VOCAB_QUESTION_FILE,
            self.cfg.MODEL.T_ENCODER,
            self.cfg.MODEL.T_CTRL,
            True,
            self.cfg.VOCAB_ANSWER_FILE,
            self.cfg.VOCAB_LAYOUT_FILE,
            self.cfg.MODEL.H_IMG,
            self.cfg.MODEL.W_IMG,
        )
        loc = PreprocessedClevr(
            filename_loc,
            self.cfg.VOCAB_QUESTION_FILE,
            self.cfg.MODEL.T_ENCODER,
            self.cfg.MODEL.T_CTRL,
            True,
            self.cfg.VOCAB_ANSWER_FILE,
            self.cfg.VOCAB_LAYOUT_FILE,
            self.cfg.MODEL.H_IMG,
            self.cfg.MODEL.W_IMG,
        )
        return ConcatDataset(clevr, loc)

    def setup(self, stage=None):
        dataset = self.cfg.DATASET
        assert dataset in ["vqa", "loc", "joint"]
        if dataset == "vqa":
            self.clevr_train = self._construct_clevr_(self.cfg.TRAIN_IMDB_FILE)
            self.clevr_val = self._construct_clevr_(self.cfg.VAL_IMDB_FILE)
            self.clevr_test = self._construct_clevr_(self.cfg.TEST_IMDB_FILE)
            self.clevr_module = self.clevr_train
            noop_idx = self.clevr_train.layout_dict.word2idx("_NoOp")
            self.collate = lambda x: clevr_collate(x, self.cfg.MODEL.T_CTRL, noop_idx)
        elif dataset == "loc":
            self.clevr_train = self._construct_clevr_(self.cfg.TRAIN_LOC_IMDB_FILE)
            self.clevr_val = self._construct_clevr_(self.cfg.VAL_LOC_IMDB_FILE)
            self.clevr_test = self._construct_clevr_(self.cfg.TEST_LOC_IMDB_FILE)
            self.clevr_module = self.clevr_train
            noop_idx = self.clevr_train.layout_dict.word2idx("_NoOp")
            self.collate = lambda x: clevr_collate(x, self.cfg.MODEL.T_CTRL, noop_idx)
        else:
            self.clevr_train = self._construct_joint_(
                self.cfg.TRAIN_IMDB_FILE, self.cfg.TRAIN_LOC_IMDB_FILE
            )
            self.clevr_val = self._construct_joint_(
                self.cfg.VAL_IMDB_FILE, self.cfg.VAL_LOC_IMDB_FILE
            )
            self.clevr_test = self._construct_joint_(
                self.cfg.TEST_IMDB_FILE, self.cfg.TEST_LOC_IMDB_FILE
            )
            self.clevr_module = self.clevr_train
            noop_idx = self.clevr_train.layout_dict.word2idx("_NoOp")
            self.collate = lambda x: joint_collate(x, self.cfg.MODEL.T_CTRL, noop_idx)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        clevr_train = DataLoader(
            self.clevr_train, batch_size=self.batch_size, num_workers=8, pin_memory=True, collate_fn=self.collate
        )
        return clevr_train

    def val_dataloader(self):
        clevr_val = DataLoader(
            self.clevr_val, num_workers=8, pin_memory=True, batch_size=10 * self.batch_size, collate_fn=self.collate
        )
        return clevr_val

    def test_dataloader(self):
        clevr_test = DataLoader(
            self.clevr_test, num_workers=8, pin_memory=True, batch_size=10 * self.batch_size, collate_fn=self.collate
        )
        return clevr_test
