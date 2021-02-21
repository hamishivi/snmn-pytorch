import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from clevr import ClevrDataModule
from clevr_model import ClevrModel
from clevr_joint_model import ClevrJointModel
from config import cfg

parser = argparse.ArgumentParser(
    description="Test the SNMN with a given saved model and config. Make sure the config and saved model match!"
)
parser.add_argument(
    "config",
    type=str,
    nargs="?",
    default=None,
    help="config yaml file, see configs folder for examples to use.",
)
parser.add_argument(
    "model", type=str, nargs="?", default=None, help="saved model file."
)

args = parser.parse_args()

# set seed for repro
pl.seed_everything(42)

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

model = model_to_load.load_from_checkpoint(
    args.model,
    cfg=cfg,
    num_choices=num_choices,
    module_names=module_names,
    num_vocab=vocab_size,
    img_sizes=img_sizes,
)

logger = CSVLogger("logs", name=f"{args.config}_dev")

trainer = pl.Trainer(
    gpus=-1,
    gradient_clip_val=cfg.TRAIN.GRAD_MAX_NORM,
    progress_bar_refresh_rate=20,
    reload_dataloaders_every_epoch=True,
    max_steps=cfg.TRAIN.MAX_ITER,
    logger=logger,
    deterministic=True,
)

trainer.fit(model, clevr.val_dataloader())
