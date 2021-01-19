import re
import io

import skimage.io
import skimage.transform
import numpy as np
import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import base64

from clevr_model import ClevrModel
from clevr_joint_model import ClevrJointModel
from utils import VocabDict, sequence_mask, batch_feat_grid2bbox


channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
H = 224
W = 224
_SENTENCE_SPLIT_REGEX = re.compile(r"(\W+)")


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


def tokenize(sentence):
    tokens = _SENTENCE_SPLIT_REGEX.split(sentence.lower())
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def _att_softmax(att):
    exps = np.exp(att - np.max(att))
    softmax = exps / np.sum(exps)
    return softmax


def attention_interpolation(im, att):
    softmax = _att_softmax(att)
    att_reshaped = skimage.transform.resize(softmax, im.shape[:2], order=3)
    # normalize the attention
    # make sure the 255 alpha channel is at least 3x uniform attention
    att_reshaped /= np.maximum(np.max(att_reshaped), 3.0 / att.size)
    att_reshaped = att_reshaped[..., np.newaxis]

    # make the attention area brighter than the rest of the area
    vis_im = att_reshaped * im + (1 - att_reshaped) * im * 0.45
    vis_im = vis_im.astype(im.dtype)
    return vis_im


def predict_sample(cfg, checkpoint_filename, question_text, image_file):
    answer_dict = VocabDict(cfg.VOCAB_ANSWER_FILE)
    layout_dict = VocabDict(cfg.VOCAB_LAYOUT_FILE)
    question_dict = VocabDict(cfg.VOCAB_QUESTION_FILE)
    num_choices = answer_dict.num_vocab
    module_names = layout_dict.word_list
    vocab_size = question_dict.num_vocab
    img_sizes = (
        H,
        W,
        cfg.MODEL.H_IMG,
        cfg.MODEL.W_IMG,
        cfg.MODEL.H_IMG * 1.0 / H,
        cfg.MODEL.W_IMG * 1.0 / W,
    )
    with torch.no_grad():
        # preprocessing: extract features.
        resnet101 = models.resnet101(pretrained=True)
        resnet101_c4 = ResnetC4(resnet101)
        im = skimage.io.imread(image_file)[..., :3]
        assert im.dtype == np.uint8
        im = skimage.transform.resize(im, [H, W], preserve_range=True)
        im_val = im[np.newaxis, ...] - channel_mean
        im_val = np.swapaxes(im_val, 3, 1)
        im_val = np.swapaxes(im_val, 2, 3)
        resnet101_c4_val = resnet101_c4(torch.tensor(im_val, dtype=torch.float))
        resnet101_c4_val = resnet101_c4_val.permute([0, 2, 3, 1])
        # prepro: tokenize
        question_tokens = tokenize(question_text)
        # prepro: turn everything into the indices required
        question_inds = [question_dict.word2idx(w) for w in question_tokens]
        seq_length = len(question_inds)
        image_feat = resnet101_c4_val
        # load model
        if cfg.MODEL.BUILD_LOC and cfg.MODEL.BUILD_VQA:
            model_to_load = ClevrJointModel
        else:
            model_to_load = ClevrModel
        model = model_to_load.load_from_checkpoint(
            checkpoint_filename,
            cfg=cfg,
            num_choices=num_choices,
            module_names=module_names,
            num_vocab=vocab_size,
            img_sizes=img_sizes,
        )
        # get output
        question_inds = torch.tensor(question_inds).unsqueeze(0)
        seq_length = torch.tensor(seq_length).unsqueeze(0)
        output = model(question_inds, sequence_mask(seq_length), image_feat)
        # return output.
        # image
        answer_output = {
            "module_probs": F.softmax(output["module_logits"][0], 1).cpu().numpy(),
            "module_dict": layout_dict,
            "qattns": output["qattns"][0].cpu().numpy().tolist(),
            "qtokens": question_tokens,
            "iattns": output["iattns"].cpu().numpy(),
        }
        # image attn vis
        attn_imgs = []
        for i in range(output["iattns"].size(1)):
            a_img = attention_interpolation(im, output["iattns"][0, i].numpy())
            a_img = Image.fromarray(a_img.astype("uint8"))
            rawBytes = io.BytesIO()
            a_img.save(rawBytes, "JPEG")
            rawBytes.seek(0)
            a_img_base64 = base64.b64encode(rawBytes.read())
            attn_imgs.append(a_img_base64)

        answer_output["iattns"] = attn_imgs
        if cfg.MODEL.BUILD_VQA:
            answer_output["answer_probs"] = (
                F.softmax(output["logits"][0], 0).cpu().numpy()
            )
            answer_output["answer_dict"] = answer_dict
        if cfg.MODEL.BUILD_LOC:
            feat_h, feat_w, _, _, stride_h, stride_w = img_sizes
            bbox_pred = batch_feat_grid2bbox(
                torch.argmax(output["loc_scores"], 1),
                output["bbox_offset"],
                stride_h,
                stride_w,
                feat_h,
                feat_w,
            )
            answer_output["bbox_pred"] = bbox_pred
        return answer_output


if __name__ == "__main__":
    from config import cfg

    cfg.merge_from_file("configs/vqa_scratch.yaml")
    cfg.freeze()
    print(
        predict_sample(
            cfg,
            "What size is the cylinder that is left of the brown metal thing that is left of the big sphere?",
            "/Users/hamishivison/Downloads/test_image.jpg",
        )
    )
