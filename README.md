# Performance of onset detection using neural networks

Supplementary code to my thesis. The purpose of this repository is to
make it easy for others to reproduce the results that I've reported in
my thesis.

## Prerequisites

Keras, tensorflow and [madmom](https://github.com/CPJKU/madmom). These
can all be installed using pip.

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
