########################################################################
# Preprocessing
# =============
# This file contains functions for preprocessing the audio data.
########################################################################
from collections import namedtuple
from config import get_config
from hashlib import sha256
from os import listdir
from os.path import basename, exists, join, splitext
from pickle import dump, load
from random import Random
import numpy as np

# AudioSample is a 4-tuple representing a preprocessed audio
# sample. The attributes are as follows:
#
# x: A numpy array where each item in the array represents 10 ms of
# audio.
#
# y: A numpy array of annotations. 1 if the corresponding 10 ms of
# audio contains an onset and 0 otherwise.
#
# a: A numpy array containing the onsets time offsets.
#
# name: Basename of the audio file.
AudioSample = namedtuple('AudioSample', ['x', 'y', 'a', 'name'])

def list_audio_files(data_dir):
    audio_dir = join(data_dir, 'audio')
    return [join(audio_dir, f) for f in sorted(listdir(audio_dir))]

def list_annotation_files(data_dir):
    ann_dir = join(data_dir, 'annotations', 'onsets')
    return [join(ann_dir, f) for f in sorted(listdir(ann_dir))]

def process_files_decorated(files, fun):
    fmt = '[%3d/%3d] %-90s '
    n = len(files)
    for i, f in enumerate(files):
        args = (i + 1, n, basename(f))
        print(fmt % args, end = '', flush = True)
        yield fun(f)
        print('DONE')

def preprocess_data(nn, seed, dataset_dir):
    audio_files = list_audio_files(dataset_dir)
    ann_files = list_annotation_files(dataset_dir)

    X = list(process_files_decorated(audio_files, nn.preprocess_x))
    A = [np.loadtxt(f) for f in ann_files]
    Y = [nn.preprocess_y(a, len(x)) for (a, x) in zip(A, X)]
    N = [splitext(basename(n))[0] for n in audio_files]
    D = [AudioSample(x, y, a, n) for x, y, a, n in zip(X, Y, A, N)]
    Random(seed).shuffle(D)

    # I'm not sure about this...
    if nn.__name__ == 'cnn':
        print('* CNN requires standardized features.')
        all_data = np.concatenate([d.x for d in D])
        mean = np.mean(all_data, axis = 0)
        std = np.std(all_data, axis = 0)
        D = [AudioSample((d.x - mean) / std, d.y, d.a, d.name) for d in D]
    return D

# Checks that the hexdigest matches. Good for verifying that cached
# data is correct.
def check_digest(expected_digest, cache_file):
    with open(cache_file, 'rb') as f:
        h = sha256()
        h.update(f.read())
        real_digest = h.hexdigest()
        if expected_digest != real_digest:
            fmt = "Digest mismatch, expected '%s', got '%s'"
            raise Exception(fmt % (expected_digest, real_digest))

# Generates the cache by calling fun if it doesn't already exist.
def load_cached_data(cache, fun):
    loading_fmt = '* Loading data from pickle cache %s...'
    if exists(cache):
        print(loading_fmt % cache)
        return load(open(cache, 'rb'))
    generating_fmt = '* Generating pickle cache %s...'
    print(generating_fmt % cache)
    data = fun()
    dump(data, open(cache, 'wb'), protocol = 2)
    return data

def load_data(nn_type):
    config = get_config()
    digest = config[nn_type]['digest']
    seed = config[nn_type]['seed']
    cache_dir = config['cache-dir']
    data_dir = config['data-dir']
    nn = config[nn_type]['module']

    cache_file = join(cache_dir, 'cache-%s.pkl' % nn_type)
    fun = lambda: preprocess_data(nn, seed, data_dir)
    D = load_cached_data(cache_file, fun)
    check_digest(digest, cache_file)
    return nn, [D[i::8] for i in range(8)]
