import sys

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from clevr import ClevrDataModule
from clevr_model import ClevrModel
from config import cfg

# set seed for repro
pl.seed_everything(42)

# cfg prepro (todo: use argparse instead of argv)
if len(sys.argv) >= 2:
    cfg.merge_from_file(sys.argv[1])  # path to a valid cfg to use
cfg.freeze()

clevr = ClevrDataModule(cfg, cfg.TRAIN.BATCH_SIZE)
clevr.setup()

# really these should be exposed in the data module...
num_choices = clevr.clevr_module.get_answer_choices()
module_names = clevr.clevr_module.get_module_names()
vocab_size = clevr.clevr_module.get_vocab_size()


if cfg.LOAD:
    model = ClevrModel.load_from_checkpoint(
        cfg.CHECKPOINT_FILENAME,
        cfg=cfg,
        num_choices=num_choices,
        module_names=module_names,
        num_vocab=vocab_size,
    )
else:
    model = ClevrModel(cfg, num_choices, module_names, vocab_size)

wandb_logger = WandbLogger(project=cfg.WANDB_PROJECT_NAME, log_model=True)

trainer = pl.Trainer(
    gpus=-1,
    gradient_clip_val=cfg.TRAIN.GRAD_MAX_NORM,
    progress_bar_refresh_rate=20,
    reload_dataloaders_every_epoch=True,
    max_steps=cfg.TRAIN.MAX_ITER,
    logger=wandb_logger,
    deterministic=True,
)

trainer.fit(model, clevr)

trainer.test(datamodule=clevr, ckpt_path=None)
