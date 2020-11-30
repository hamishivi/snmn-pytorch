import sys

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from clevr import PreprocessedClevr
from model import Model
from config import cfg

cfg.merge_from_file(sys.argv[1])  # path to a valid cfg to use
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
    cfg.MODEL.W_FEAT,
)
val_dataset = PreprocessedClevr(
    cfg.VAL_IMDB_FILE,
    cfg.VOCAB_QUESTION_FILE,
    cfg.MODEL.T_ENCODER,
    cfg.MODEL.T_CTRL,
    True,
    cfg.VOCAB_ANSWER_FILE,
    cfg.VOCAB_LAYOUT_FILE,
    cfg.MODEL.H_FEAT,
    cfg.MODEL.W_FEAT,
)

num_choices = train_dataset.get_answer_choices()
module_names = train_dataset.get_module_names()
vocab_size = train_dataset.get_vocab_size()

model = Model(cfg, num_choices, module_names, vocab_size)
trainer = pl.Trainer(
    gpus=1,
    gradient_clip_val=cfg.TRAIN.GRAD_MAX_NORM,
    progress_bar_refresh_rate=20,
    reload_dataloaders_every_epoch=True,
    max_steps=cfg.TRAIN.MAX_ITER,
)
trainer.fit(
    model,
    DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=4, pin_memory=True
    ),
    DataLoader(
        val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=4, pin_memory=True
    ),
)
