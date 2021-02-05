import os
from datetime import datetime
from glob import glob
import argparse

import skimage.io
import skimage.transform
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

batch_size = 64

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


resnet101 = models.resnet101(pretrained=True)


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.im_paths = image_paths

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        impath = self.im_paths[idx]
        im = skimage.io.imread(impath)[..., :3]
        assert im.dtype == np.uint8
        im = skimage.transform.resize(im, [H, W], preserve_range=True)
        im_val = im - channel_mean
        # pytorch uses channels-first, so we reorder stuff to fit this.
        im_val = np.swapaxes(im_val, 2, 0)  # [H,W,C] -> [C,W,H]
        im_val = np.swapaxes(im_val, 1, 2)  # [C,W,H] -> [C,H,W]
        return impath, torch.tensor(im_val, dtype=torch.float)


def batched_extract(resnet101_c4, vals):
    resnet101_c4_val = resnet101_c4(vals).cpu()
    resnet101_c4_val = resnet101_c4_val.permute([0, 2, 3, 1])
    return resnet101_c4_val


def extract_dataset_resnet101_c4(device, image_dir, save_dir, ext_filter="*.png"):
    resnet101_c4 = ResnetC4(resnet101).to(device)
    image_list = glob(image_dir + "/" + ext_filter)
    os.makedirs(save_dir, exist_ok=True)
    imdataset = ImageDataset(image_list)
    imloader = DataLoader(imdataset, num_workers=2, batch_size=batch_size)
    with torch.no_grad():
        for impaths, imvals in tqdm(imloader):
            res = batched_extract(resnet101_c4, imvals.to(device))
            for i, path in enumerate(impaths):
                image_name = os.path.basename(path).split(".")[0]
                save_path = os.path.join(save_dir, image_name + ".npy")
                np.save(save_path, res[i].numpy())


def main():
    dev = torch.device("cuda")  # assume cuda available
    image_sets = ["train", "val", "test"]
    if args.loc:
        image_sets = ["loc_train", "loc_val", "loc_test"]
    for image_set in image_sets:
        now = datetime.now()
        print(f'{now.strftime("%H:%M:%S")}: extracting {image_set}')
        extract_dataset_resnet101_c4(
            dev,
            os.path.join(image_basedir, image_set),
            os.path.join(save_basedir, image_set),
        )


if __name__ == "__main__":
    main()
