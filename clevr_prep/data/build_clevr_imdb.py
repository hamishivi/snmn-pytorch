import numpy as np
import json
import os
import re
import argparse

import sys
from tqdm import tqdm

sys.path.append("../../")  # NOQA

question_file = "./CLEVR_%s_questions_gt_layout.json"
image_dir = "../clevr_dataset/images/%s/"
feature_dir = "./resnet101_c4/%s/"

_SENTENCE_SPLIT_REGEX = re.compile(r"(\W+)")


def tokenize(sentence):
    tokens = _SENTENCE_SPLIT_REGEX.split(sentence.lower())
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def build_imdb(image_set):
    print("building imdb %s" % image_set)
    with open(question_file % image_set) as f:
        questions = json.load(f)["questions"]
    abs_image_dir = os.path.abspath(image_dir % image_set)
    abs_feature_dir = os.path.abspath(feature_dir % image_set)
    imdb = [None] * len(questions)
    for n_q, q in tqdm(enumerate(questions)):
        image_name = q["image_filename"].split(".")[0]
        image_path = os.path.join(abs_image_dir, q["image_filename"])
        feature_path = os.path.join(abs_feature_dir, image_name + ".npy")
        question_str = q["question"]
        question_tokens = tokenize(question_str)
        bbox = q["bbox"] if "bbox" in q else None
        gt_layout_tokens = q["gt_layout"] if "gt_layout" in q else None

        iminfo = dict(
            image_name=image_name,
            image_path=image_path,
            feature_path=feature_path,
            question_str=question_str,
            question_tokens=question_tokens,
            bbox=bbox,
            gt_layout_tokens=gt_layout_tokens,
        )
        imdb[n_q] = iminfo
    return imdb


parser = argparse.ArgumentParser(
    description="Extract construct module ground truth layouts for clevr or clevr-ref."
)
parser.add_argument(
    "--loc",
    action="store_true",
    help="if used, extract features from clevr-ref dataset rather than regular clevr.",
)
args = parser.parse_args()

train, val, test = "train", "val", "test"
if args.loc:
    train = "loc_" + train
    val = "loc_" + val
    test = "loc_" + test

imdb_trn = build_imdb(train)
imdb_val = build_imdb(val)
imdb_tst = build_imdb(test)

os.makedirs("./imdb", exist_ok=True)
np.save(f"./imdb/imdb_{train}.npy", np.array(imdb_trn))
np.save(f"./imdb/imdb_{val}.npy", np.array(imdb_val))
np.save(f"./imdb/imdb_{test}.npy", np.array(imdb_tst))
