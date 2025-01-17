import sys
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from clevr import ClevrDataModule
from clevr_model import ClevrModel, init_weights
from clevr_joint_model import ClevrJointModel
from config import cfg


# argparse. Currently two arguments: the config and the seed.
parser = argparse.ArgumentParser(
    description="Train the SNMN with a given config file and seed."
)
parser.add_argument(
    "config",
    type=str,
    default=None,
    help="config yaml file, see configs folder for examples to use.",
)
parser.add_argument(
    "-s",
    "--seed",
    type=int,
    nargs="?",
    default=42,
    help="seed to use for reproducibility.",
)

args = parser.parse_args()

# set seed for repro
pl.seed_everything(args.seed)

if args.config:
    cfg.merge_from_file(args.config)  # path to a valid cfg to use
cfg.freeze()

clevr = ClevrDataModule(cfg, cfg.TRAIN.BATCH_SIZE)
clevr.setup()

# really these should be exposed in the data module...
num_choices = clevr.clevr_module.get_answer_choices()
module_names = clevr.clevr_module.get_module_names()
vocab_size = clevr.clevr_module.get_vocab_size()
img_sizes = clevr.clevr_module.get_img_sizes()

if cfg.MODEL.BUILD_LOC and cfg.MODEL.BUILD_VQA:
    model_to_load = ClevrJointModel
else:
    model_to_load = ClevrModel

if cfg.LOAD:
    model = model_to_load.load_from_checkpoint(
        cfg.CHECKPOINT_FILENAME,
        cfg=cfg,
        num_choices=num_choices,
        module_names=module_names,
        num_vocab=vocab_size,
        img_sizes=img_sizes,
    )
else:
    model = model_to_load(cfg, num_choices, module_names, vocab_size, img_sizes)
    model.apply(init_weights)

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
