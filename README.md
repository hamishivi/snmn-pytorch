# SNMN-pytorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eGroHrcsGu7OVIqexjouHzV0ETBu8dV0?usp=sharing)
![Heroku](https://pyheroku-badge.herokuapp.com/?app=snmn-pytorch&style=flat)

A pytorch adaption of the stack neural module network. **not an official repository**, but rather just something whipped up during a student's free time. Code largely copied from the original [snmn repository](https://github.com/ronghanghu/snmn), with the obvious changes made to make use of pytorch and pytorch-lightning.

Check out the colab link above to train the model yourself, with preprocessing included. Otherwise, continue reading to see how to do it yourself. Check out a demo of the model [here](https://snmn-pytorch.herokuapp.com/), and my own blog post on this project [here](https://hamishivi.github.io)! 😊

**Note: I'm still playing slightly with this repository due to VQA results not being quite up to par. I may make updates now and then.**

## Usage - Training

### Packages

Make sure to install python and install the relevant packages (`dev-requirements.txt` contains the requirements for training and development):
```bash
pip install -r dev-requirements.txt
```

Then login to [wandb](wandb.ai) for logging purposes:
```bash
wandb login
```

I use pytorch-lightning for training, so you can swap to other logging solutions by modifying `train.py` fairly easily.

### Download and Preprocess Data

Note that theoretically you can save the results from this step, but the outputs are large (> 70gig), so I have not done this. This is hard-coded to use a single gpu currently.

#### CLEVR Dataset

1. Download the CLEVR dataset from http://cs.stanford.edu/people/jcjohns/clevr/, and symlink it to `clevr_prep/clevr_dataset`. After this step, the file structure should look like

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
python extract_resnet101_c4.py # feature extraction
python get_ground_truth_layout.py  # construct expert policy
python build_clevr_imdb.py  # build image collections
cd ../../
```

#### CLEVR-ref Dataset

The CLEVR-ref dataset is processed in largely the same way, but uses a different dataset.

1. First, download it from http://people.eecs.berkeley.edu/~ronghang/projects/snmn/CLEVR_loc.tgz, and then symlink it to `clevr_prep/clevr_loc_dataset`. After this step, the file structure should look like:
```
exp_clevr_snmn/clevr_loc_dataset/
  images/
    loc_train/
      CLEVR_loc_train_000000.png
      ...
    loc_val/
    loc_test/
  questions/
    CLEVR_loc_train_questions.json
    CLEVR_loc_val_questions.json
    CLEVR_loc_test_questions.json
  ...
```

2. Again, extract the features from the images and store them on disk. Again, we re-use the code from the original SNMN repository, so you should be able to use these features for that code and vice-versa.

```bash
cd ./clevr_prep/data/
python extract_resnet101_c4.py --loc
python get_ground_truth_layout.py  --loc
python build_clevr_imdb.py  --loc
cd ../../
```

### Train!

At this point, we can train! Simply run
```bash
python train.py configs/<config name>
```

Where the config file is one of the files present in the `configs` directory. Look below for short explanations on each config and expected performance on each.  **Note that you'll need to have downloaded the regular CLEVR dataset for the VQA and joint configs, and the CLEVR-ref dataset for the loc and joint configs.** Feel free to make your own config yamls to investigate different hyperparameters and such! We also use pytorch-lightning to handle training and logging, so take a look at `train.py` and `config.py` to see what training options are used to tune them to your preference.

### Testing

To test a saved checkpoint, run
```bash
python test.py configs/<config name> <checkpoint file>
```

Make sure the config and saved model match, and you'll need to have the appropriate dataset downloaded. By default, this will save a csv into a folder called `results`.

## Results

As this is a re-implementation, performance is not exactly the same as reported in the original repository. See the table below for the expected performance on our implementation and the original implementation for each provided config file. For the joint configs (which have two metrics as they jointly train on two tasks), we provide the VQA (aka regular CLEVR) accuracy first, and then the CLEVR-ref accuracy. Note that the scratch accuracy is a bit low, potentially due to a lack of fine-tuning and testing (many runs get low performance both in original and this repo - performance is best over a few runs). I found that turning on module validation helped performance - but the network appeared to just rely on the 'find' module for everything (indicating that maybe it wasn't actually using modules as intended).

| Config | Description | Original Performance | Our Performance |
| ------ | ----------- | -------------------- | --------------- |
| `vqa_scratch` | regular CLEVR dataset, training on final answers. | 93.0% | 89.5% |
| `vqa_gt_layout` | regular CLEVR dataset, training on final answers and ground truth module layouts. | 96.6% | 96.1% |
| `loc_scratch` | CLEVR-ref dataset, training on bounding boxes. | 93.4% | 88.7% |
| `loc_gt_layout` | CLEVR-ref dataset, training on bounding boxes and ground truth module layouts. | 96.0% | 93.2% |
| `joint_scratch` | CLEVR + CLEVR-ref together, training on final answers and bounding boxes.  (vqa/loc) | 93.9% / 95.4% | 90.4% / 81.4% |
| `joint_gt_layout` | CLEVR + CLEVR-ref together, training on answers, bounding boxes, and ground truth module layouts.  (vqa/loc) | 96.5% / 96.2% | 95.9% / 95.7% |


## Demo

You can see a visualisation of the visual question answering task (with modules visualised) [here](https://snmn-pytorch.herokuapp.com/). Note that it's a free heroku app I setup, so the first inference run takes a fair bit of time to run while it downloads the pretrained models. For more details, check out [my blog post on this project](https://hamishivi.github.io).

You should be able to run the app yourself by running `uvicorn server.main:app`. Check out [uvicorn's options](https://www.uvicorn.org/#usage) for details on how to set specific ports etc. The server will download the pretrained models required for inference when first needed, so make sure you have ~300MB free for the three different pretrained models used.
