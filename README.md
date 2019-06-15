# Performance of onset detection using neural networks

Supplementary code to my thesis. The purpose of this repository is to
make it easy for others to reproduce the results that I've reported in
my thesis.

## Prerequisites

The Python packages Keras, tensorflow
and [madmom](https://github.com/CPJKU/madmom). These can all be
installed using pip. ffmpeg is also needed. If it isn't already
installed it can be installed using:
```
$ cd /tmp && wget https://ffmpeg.org/releases/ffmpeg-4.1.tar.bz2 \
    && tar xvjf ffmpeg-4.1.tar.bz2 && cd ffmpeg-4.1 \
    && ./configure && make
$ export PATH=/tmp/ffmpeg-4.1:$PATH
```

## The Böck dataset

First the training dataset has to be downloaded. This is done using
the `download.py` script:
```
$ python download.py /tmp/
* Downloading document with id 1ICEfaZ2r_cnqd3FLNC5F_UOEUalgV7cv to /tmp/onsets.zip.
* Extracting /tmp/onsets.zip
```
The script hardcodes the dataset location
to
[this](https://drive.google.com/file/d/1ICEfaZ2r_cnqd3FLNC5F_UOEUalgV7cv/view) url. If
it ever changes then the `DOC_ID` constant in the script needs to be
updated.

## Configuration

Paths to input, output and cache data has to be configured by
modifying the `CONFIGS` constant in the `config.py` file. The right
config is selected during runtime by matching on the system and
hostname. This way the same `config.py` can be used on multiple
systems without requiring any changes.

The `data-dir` field should be set to the directory containing the
Böck dataset, `cache-dir` to a directory storing cache files in pickle
format and `model-dir` to the directory in which built models should
be stored.

The `seed` field contains the seed to the random number generators
ensuring that *exactly* the same results a produced every
time. `digest` contains the checksum of the cache file. It is
important that the cache file does not change during training or
evaluation.

## Training

Training is done using the `main.py` script:
```
$ python main.py -t 0:8 -n rnn --epochs 20
```
The above command would train the eight folds using the recurrent
neural network (rnn) architecture for 20 epochs each.

## Evaluation

Evaluation is done using the `main.py` script:
```
$ python main.py -e 0:1 -n rnn
...
sum for 41 files
  #:   3368 TP:   2861 FP:   387 FN:   507
  Prec: 0.881 Rec: 0.849 F-score: 0.865
```
The above command evaluates the first fold (with index 0) of the rnn
architecture.
