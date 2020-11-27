'''
DataLoader class for CLEVR used in training.
'''
from torch.utils.data import Dataset
from utils import VocabDict
import numpy as np

class PreprocessedClevr(Dataset):
    def __init__(self, imdb_file, vocab_question_file, t_encoder, t_decoder, load_gt_layout, vocab_answer_file, vocab_layout_file, img_h, img_w, prune_filter_module=True,):
        self.imdb = np.load(imdb_file, allow_pickle=True)
        self.vocab_dict = VocabDict(vocab_question_file)
        self.t_encoder = t_encoder
        # peek one example to see whether answer and gt_layout are in the data
        self.load_answer = ('answer' in self.imdb[0] and self.imdb[0]['answer'] is not None)
        self.load_bbox = ('bbox' in self.imdb[0] and self.imdb[0]['bbox'] is not None)
        self.load_gt_layout = (load_gt_layout and ('gt_layout_tokens' in self.imdb[0] and self.imdb[0]['gt_layout_tokens'] is not None))
        # load answer dict
        self.answer_dict = VocabDict(vocab_answer_file)
        self.t_decoder = t_decoder
        self.layout_dict = VocabDict(vocab_layout_file)
        if self.load_gt_layout:
            self.prune_filter_module = prune_filter_module
        # load one feature map to peek its size
        feats = np.load(self.imdb[0]['feature_path'])
        self.feat_H, self.feat_W, self.feat_D = feats.shape[1:]
        if self.load_bbox:
            self.img_H = img_h
            self.img_W = img_w
            self.stride_H = self.img_H * 1. / self.feat_H
            self.stride_W = self.img_W * 1. / self.feat_W

    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):
        iminfo = self.imdb[idx]
        question_inds = [self.vocab_dict.word2idx(w) for w in iminfo['question_tokens']]
        seq_length = len(question_inds)
        image_feat = np.load(iminfo['feature_path'])
        image_path = iminfo['image_path']
        if self.load_answer:
            answer_idx = self.answer_dict.word2idx(iminfo['answer'])
        if self.load_bbox:
            bbox_batch = iminfo['bbox']
            bbox_ind, bbox_offset = bbox2feat_grid(iminfo['bbox'], self.stride_H, self.stride_W, self.feat_H, self.feat_W)
        if self.load_gt_layout:
            gt_layout_tokens = iminfo['gt_layout_tokens']
            if self.prune_filter_module:
                # remove duplicated consequtive modules
                # (only keeping one _Filter)
                for n_t in range(len(gt_layout_tokens)-1, 0, -1):
                    if (gt_layout_tokens[n_t-1] in {'_Filter', '_Find'}
                            and gt_layout_tokens[n_t] == '_Filter'):
                        gt_layout_tokens[n_t] = None
                gt_layout_tokens = [t for t in gt_layout_tokens if t]
            layout_inds = [self.layout_dict.word2idx(w) for w in gt_layout_tokens]

        batch = [question_inds, seq_length, image_feat, image_path]
        if self.load_answer:
            batch.append(answer_idx)
        if self.load_bbox:
            batch.append(bbox_batch)
            batch.append(bbox_ind)
            batch.append(bbox_offset)
        if self.load_gt_layout:
            batch.append(layout_inds)
        return batch

    def get_answer_choices(self):
        return self.answer_dict.num_vocab

    def get_module_names(self):
        return self.layout_dict.word_list

def bbox2feat_grid(bbox, stride_H, stride_W, feat_H, feat_W):
    x1, y1, w, h = bbox
    x2 = x1 + w - 1
    y2 = y1 + h - 1
    # map the bbox coordinates to feature grid
    x1 = x1 * 1. / stride_W - 0.5
    y1 = y1 * 1. / stride_H - 0.5
    x2 = x2 * 1. / stride_W - 0.5
    y2 = y2 * 1. / stride_H - 0.5
    xc = min(max(int(round((x1 + x2) / 2.)), 0), feat_W - 1)
    yc = min(max(int(round((y1 + y2) / 2.)), 0), feat_H - 1)
    ind = yc * feat_W + xc
    offset = x1 - xc, y1 - yc, x2 - xc, y2 - yc
    return ind, offset