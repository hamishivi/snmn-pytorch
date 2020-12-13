import os
import sys
from glob import glob
import argparse

import skimage.io
import skimage.transform
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from tqdm import tqdm

channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

parser = argparse.ArgumentParser(
    description="Extract conv features for clevr or clevr-ref"
)
parser.add_argument(
    "--loc",
    action="store_true",
    help="if used, extract features from clevr-ref dataset rather than regular clevr.",
)


args = parser.parse_args()

if args.loc:
    image_basedir = "../clevr_loc_dataset/images/"
else:
    image_basedir = "../clevr_dataset/images/"

save_basedir = "./resnet101_c4/"
H = 224
W = 224


# class to extract the fourth filter vals out of resnet.
class ResnetC4(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.features = nn.Sequential(
            # stop at conv4
            *list(resnet.children())[:-3]
        )

    def forward(self, x):
        x = self.features(x)
        return x


# we assume there's a gpu on offer and a single gpu at that.
resnet101 = models.resnet101(pretrained=True)
resnet101_c4 = ResnetC4(resnet101).cuda()


def extract_image_resnet101_c4(impath):
    with torch.no_grad():
        im = skimage.io.imread(impath)[..., :3]
        assert im.dtype == np.uint8
        im = skimage.transform.resize(im, [H, W], preserve_range=True)
        im_val = im[np.newaxis, ...] - channel_mean
        # pytorch uses channels-first, so we reorder stuff to fit this.
        im_val = np.swapaxes(im_val, 3, 1)  # [B,H,W,C] -> [B,C,W,H]
        im_val = np.swapaxes(im_val, 2, 3)  # [B,C,W,H] -> [B,C,H,W]
        resnet101_c4_val = resnet101_c4(
            torch.tensor(im_val, device="cuda", dtype=torch.float)
        )
        # reorder again
        resnet101_c4_val = resnet101_c4_val.permute([0, 2, 3, 1])
        return resnet101_c4_val.cpu().numpy()


def extract_dataset_resnet101_c4(image_dir, save_dir, ext_filter="*.png"):
    image_list = glob(image_dir + "/" + ext_filter)
    os.makedirs(save_dir, exist_ok=True)

    for n_im, impath in tqdm(enumerate(image_list)):
        image_name = os.path.basename(impath).split(".")[0]
        save_path = os.path.join(save_dir, image_name + ".npy")
        if not os.path.exists(save_path):
            resnet101_c4_val = extract_image_resnet101_c4(impath)
            np.save(save_path, resnet101_c4_val)


image_sets = ["train", "val", "test"]
if args.loc:
    image_sets = ["loc_train", "loc_val", "loc_test"]
for image_set in image_sets:
    print("Extracting image set " + image_set)
    extract_dataset_resnet101_c4(
        os.path.join(image_basedir, image_set), os.path.join(save_basedir, image_set)
    )
    print("Done.")
