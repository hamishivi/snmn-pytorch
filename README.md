# SNMN-pytorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aaTKkApKLeQOuRdXRtA2mSi3Z87ll8RC#scrollTo=Xm5CpLGbMc1r)

A pytorch adaption of the stack neural module network. **not an official repository**, but rather just something whipped up during a student's free time. Code largely copied from the original [snmn repository](https://github.com/ronghanghu/snmn), with the obvious changes made to make use of pytorch and pytorch-lightning.

Check out the colab link above to run the code yourself, with preparation included. Otherwise, continue reading to see how to run all the code yourself. 😊

## Preparation

### Download and Preprocess Data

1. Download the CLEVR dataset from http://cs.stanford.edu/people/jcjohns/clevr/, and symbol link it to exp_clevr_snmn/clevr_dataset. After this step, the file structure should look like

```
exp_clevr_snmn/clevr_dataset/
  images/
    train/
      CLEVR_train_000000.png
      ...
    val/
    test/
  questions/
    CLEVR_train_questions.json
    CLEVR_val_questions.json
    CLEVR_test_questions.json
  ...
```

2. Extract visual features from the images and store them on the disk. In our experiments, we extract visual features using ResNet-101 C4 block. Then, construct the "expert layout" from ground-truth functional programs, and build image collections (imdb) for CLEVR. These procedures can be down as follows.

```
./clevr_prep/tfmodel/resnet/download_resnet_v1_101.sh  # download ResNet-101

cd ./clevr_prep/data/
python extract_resnet101_c4.py  # feature extraction
python get_ground_truth_layout.py  # construct expert policy
python build_clevr_imdb.py  # build image collections
cd ../../
```