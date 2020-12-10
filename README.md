# SNMN-pytorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aaTKkApKLeQOuRdXRtA2mSi3Z87ll8RC#scrollTo=Xm5CpLGbMc1r)

A pytorch adaption of the stack neural module network. **not an official repository**, but rather just something whipped up during a student's free time. Code largely copied from the original [snmn repository](https://github.com/ronghanghu/snmn), with the obvious changes made to make use of pytorch and pytorch-lightning.

Check out the colab link above to run the code yourself, with preparation included. Otherwise, continue reading to see how to run all the code yourself. 😊

## Usage

### Packages

Make sure to install python and install the relevant packages:
```bash
pip install -r requirements.txt
```

Optionally, also login to [wandb](wandb.ai) for logging purposes:
```bash
wandb login
```

### Download and Preprocess Data

Note that theoretically you can save the results from this step, but the outputs are large (> 70gig), so I have not done this. This is hard-coded to use a single gpu currently.

1. Download the CLEVR dataset from http://cs.stanford.edu/people/jcjohns/clevr/, and symbol link it to clevr_prep/clevr_dataset. After this step, the file structure should look like

```
clevr_prep/clevr_dataset/
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

2. Extract visual features from the images and store them on the disk. In our experiments, we extract visual features using ResNet-101 C4 block. Then, construct the "expert layout" from ground-truth functional programs, and build image collections (imdb) for CLEVR. These procedures can be down as follows. *Note that if you did this for the original snmn repo, you should be able to re-use those preprocessed features, since they utilise the same code (the only difference being our use of the pretrained pytorch version of the resnet101 model).*

```bash
cd ./clevr_prep/data/
python extract_resnet101_c4.py  # feature extraction
python get_ground_truth_layout.py  # construct expert policy
python build_clevr_imdb.py  # build image collections
cd ../../
```

### Train!

At this point, we can train! Simply run
```bash
python train.py configs/<config name>
```

Where the config file is one of the files present in the `configs` directory. Look below for short explanations on each config and expected performance on each. Feel free to make your own config yamls to investigate different hyperparameters and such! We also use pytorch-lightning to handle training and logging, so take a look at `train.py` and `config.py` to see what training options are used to tune them to your preference.

## Results

As this is a re-implementation, performance is not exactly the same as reported in the original repository. See the table below for the expected performance on our implementation and the original implementation for each provided config file. For the joint configs (which have two metrics as they jointly train on two tasks), we provide the VQA (aka regular CLEVR) accuracy first, and then the CLEVR-ref accuracy.

| Config | Description | Original Performance | Our Performance |
| ------ | ----------- | -------------------- | --------------- |
| `vqa_scratch` | regular CLEVR dataset, training on final answers. | 93.0% | - |
| `vqa_gt_layout` | regular CLEVR dataset, training on final answers and ground truth module layouts. | 96.6% | - |
| `loc_scratch` | CLEVR-ref dataset, training on bounding boxes. | 93.4% | - |
| `loc_gt_layout` | CLEVR-ref dataset, training on bounding boxes and ground truth module layouts. | 96.0% | - |
| `joint_scratch` | CLEVR + CLEVR-ref together, training on final answers and bounding boxes. |  93.9% / 95.4% | - |
| `joint_gt_layout` | CLEVR + CLEVR-ref together, training on answers, bounding boxes, and ground truth module layouts. | 96.5% / 96.2% | - |


## Demo

TBA!