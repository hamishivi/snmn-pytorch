"""
Useful reusable methods that don't have another clear home.
"""
import re

import torch
import torch.nn.functional as F
import numpy as np


def pack_and_rnn(data, data_len, fn):
    embed = torch.nn.utils.rnn.pack_padded_sequence(
        data, data_len.cpu(), batch_first=True, enforce_sorted=False
    )
    out = fn(embed)
    return torch.nn.utils.rnn.pad_packed_sequence(out[0], batch_first=True)[0], out[1]


def get_positional_encoding(H, W, pe_dim, device):
    # this should probably be converted to torch stuff? maybe?
    assert pe_dim % 4 == 0, "pe_dim must be a multiply of 4 (h/w x sin/cos)"
    c_period = 10000.0 ** np.linspace(0.0, 1.0, pe_dim // 4)
    h_vec = np.tile(np.arange(0, H).reshape((H, 1, 1)), (1, W, 1)) / c_period
    w_vec = np.tile(np.arange(0, W).reshape((1, W, 1)), (H, 1, 1)) / c_period
    position_encoding = np.concatenate(
        (np.sin(h_vec), np.cos(h_vec), np.sin(w_vec), np.cos(w_vec)), axis=-1
    )
    position_encoding = position_encoding.reshape((1, H, W, pe_dim))
    return torch.tensor(position_encoding, device=device, dtype=torch.float)


_SENTENCE_SPLIT_REGEX = re.compile(r"(\W+)")


def tokenize(sentence):
    tokens = _SENTENCE_SPLIT_REGEX.split(sentence.lower())
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


class VocabDict:
    def __init__(self, vocab_file):
        self.word_list = load_str_list(vocab_file)
        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}
        self.num_vocab = len(self.word_list)
        self.UNK_idx = (
            self.word2idx_dict["<unk>"] if "<unk>" in self.word2idx_dict else None
        )

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_idx is not None:
            return self.UNK_idx
        else:
            raise ValueError(
                "word %s not in dictionary (while dictionary does"
                " not contain <unk>)" % w
            )

    def tokenize_and_index(self, sentence):
        inds = [self.word2idx(w) for w in tokenize(sentence)]
        return inds


def sequence_mask(lengths, maxlen=None, dtype=torch.long):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    return mask.type(dtype)


# pytorch doesnt have a channels last conv option, as far as I can see.
# so this is a little wrapper func (since im lazy and dont want to rewrite
# all my code to be channels-first).
def channels_last_conv(input, fn):
    output = fn(input.permute([0, 3, 1, 2]))
    return output.permute([0, 2, 3, 1])


def channels_last_conv_1d(input, fn):
    output = fn(input.permute([0, 2, 1]))
    return output.permute([0, 2, 1])


# copied directly from the original repo, this controls the scale used for the
# sharpen loss during training (which warms up over time, and only begins after
# a certain number of steps).
class SharpenLossScaler:
    def __init__(self, cfg):
        scaling_type = cfg.TRAIN.SHARPEN_LOSS_SCALING_TYPE
        if scaling_type == "warmup_scaling":
            self.warmup_begin_iter = cfg.TRAIN.SHARPEN_SCHEDULE_BEGIN
            self.warmup_end_iter = cfg.TRAIN.SHARPEN_SCHEDULE_END
        elif scaling_type == "func_scaling":
            self.scaling_func = eval(cfg.TRAIN.SHARPEN_LOSS_SCALING_FUNC)
            assert callable(self.scaling_func)
        else:
            raise ValueError("Unknown scaling_type {}".format(scaling_type))
        self.scaling_type = scaling_type

    def __call__(self, n_iter):
        if self.scaling_type == "warmup_scaling":
            return warmup_scaling(n_iter, self.warmup_begin_iter, self.warmup_end_iter)
        else:
            return self.scaling_func(n_iter)


def warmup_scaling(n_iter, begin_iter, end_iter):
    if n_iter >= end_iter:
        return 1.0
    elif n_iter < begin_iter:
        return 0.0

    return (n_iter - begin_iter) * 1.0 / (end_iter - begin_iter)
