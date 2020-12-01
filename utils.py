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
