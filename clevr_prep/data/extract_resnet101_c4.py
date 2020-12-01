import os
from glob import glob
import skimage.io
import skimage.transform
import numpy as np
import torchvision.models as models
from resnet_pytorch import ResnetC4

channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

image_basedir = "../clevr_dataset/images/"
save_basedir = "./resnet101_c4/"
H = 224
W = 224

# we assume there's a gpu on offer and a single gpu at that.
resnet101 = models.resnet101(pretrained=True)
resnet101_c4 = ResnetC4(resnet101).cuda()


def extract_image_resnet101_c4(impath):
    im = skimage.io.imread(impath)[..., :3]
    assert im.dtype == np.uint8
    im = skimage.transform.resize(im, [H, W], preserve_range=True)
    im_val = im[np.newaxis, ...] - channel_mean
    resnet101_c4_val = resnet101_c4(im_val).cpu().numpy()
    return resnet101_c4_val


def extract_dataset_resnet101_c4(image_dir, save_dir, ext_filter="*.png"):
    image_list = glob(image_dir + "/" + ext_filter)
    os.makedirs(save_dir, exist_ok=True)

    for n_im, impath in enumerate(image_list):
        if (n_im + 1) % 100 == 0:
            print("processing %d / %d" % (n_im + 1, len(image_list)))
        image_name = os.path.basename(impath).split(".")[0]
        save_path = os.path.join(save_dir, image_name + ".npy")
        if not os.path.exists(save_path):
            resnet101_c4_val = extract_image_resnet101_c4(impath)
            np.save(save_path, resnet101_c4_val)


for image_set in ["train", "val", "test"]:
    print("Extracting image set " + image_set)
    extract_dataset_resnet101_c4(
        os.path.join(image_basedir, image_set), os.path.join(save_basedir, image_set)
    )
    print("Done.")
