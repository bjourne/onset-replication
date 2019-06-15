# Performance of onset detection using neural networks

Supplementary code to my thesis. The purpose of this repository is to
make it easy for others to reproduce the results that I've reported in
my thesis.

## The Böck Dataset

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
