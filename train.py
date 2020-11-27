import sys

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from clevr import PreprocessedClevr
from model import Model
from config import cfg

cfg.merge_from_file(sys.argv[1]) # path to a valid cfg to use
cfg.freeze()

train_dataset = PreprocessedClevr(
    cfg.TRAIN_IMDB_FILE,
    cfg.VOCAB_QUESTION_FILE,
    cfg.MODEL.T_ENCODER,
    cfg.MODEL.T_CTRL,
    True,
    cfg.VOCAB_ANSWER_FILE,
    cfg.VOCAB_LAYOUT_FILE,
    cfg.MODEL.H_FEAT,
    cfg.MODEL.W_FEAT)
val_dataset = PreprocessedClevr(
    cfg.VAL_IMDB_FILE,
    cfg.VOCAB_QUESTION_FILE,
    cfg.MODEL.T_ENCODER,
    cfg.MODEL.T_CTRL,
    True,
    cfg.VOCAB_ANSWER_FILE,
    cfg.VOCAB_LAYOUT_FILE,
    cfg.MODEL.H_FEAT,
    cfg.MODEL.W_FEAT)

model = Model()
trainer = pl.Trainer()
trainer.fit(model, DataLoader(train_dataset), DataLoader(val_dataset))
