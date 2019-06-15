from keras.layers import SimpleRNN, Bidirectional, Masking, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import Sequence
from madmom.audio.filters import LogarithmicFilterbank
from madmom.audio.signal import FramedSignal, Signal
from madmom.audio.spectrogram import (_diff_frames,
                                      FilteredSpectrogram,
                                      LogarithmicSpectrogram,
                                      Spectrogram,
                                      SpectrogramDifference)
from madmom.audio.stft import ShortTimeFourierTransform
from madmom.features.onsets import RNNOnsetProcessor, peak_picking
from madmom.utils import combine_events
from os.path import join
from sys import argv
import numpy as np

def samples_in_audio_sample(d):
    return d.x[np.newaxis, ...]

class ArchSequence(Sequence):
    def __init__(self, D):
        self.D = D

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        part = self.D[idx : idx + 1]
        x = [d[0] for d in part]
        y = [d[1] for d in part]
        return np.array(x), np.array(y)[..., np.newaxis]

def model():
    m = Sequential()
    m.add(Masking(input_shape = (None, 266)))
    m.add(Bidirectional(SimpleRNN(units = 25,
                                  return_sequences = True)))
    m.add(Bidirectional(SimpleRNN(units = 25,
                                  return_sequences = True)))
    m.add(Bidirectional(SimpleRNN(units = 25,
                                  return_sequences = True)))
    m.add(Dense(units = 1, activation = 'sigmoid'))
    optimizer = SGD(lr = 0.01, clipvalue = 5, momentum = 0.9)
    m.compile(loss = 'binary_crossentropy',
              optimizer = optimizer,
              metrics = ['binary_accuracy'])
    return m

def preprocess_sig(sig, frame_size):
    frames = FramedSignal(sig, frame_size = frame_size, fps = 100)
    stft = ShortTimeFourierTransform(frames)
    filt = FilteredSpectrogram(stft, num_bands = 6)
    spec = np.log10(5*filt + 1)
    # Calculate difference spectrogram with ratio 0.25
    diff_frames = _diff_frames(0.25,
                               frame_size = frame_size,
                               hop_size = 441,
                               window = np.hanning)
    init = np.empty((diff_frames, spec.shape[1]))
    init[:] = np.inf
    spec = np.insert(spec, 0, init, axis = 0)
    diff_spec = spec[diff_frames:] - spec[:-diff_frames]
    np.maximum(diff_spec, 0, out = diff_spec)
    diff_spec[np.isinf(diff_spec)] = 0
    diff_spec = np.hstack((spec[diff_frames:], diff_spec))
    return diff_spec

def preprocess_x(filename):
    sig = Signal(filename, sample_rate = 44100, num_channels = 1)
    frame_sizes = [1024, 2048, 4096]
    D = [preprocess_sig(sig, fs) for fs in frame_sizes]
    return np.hstack(D)

def preprocess_y(anns, n_frames):
    y = anns[:np.searchsorted(anns, (n_frames - 0.5) / 100)]
    q = np.zeros(n_frames)
    idx = np.unique(np.round(y * 100).astype(np.int))
    q[idx] = 1
    return q

def postprocess_y(y):
    onsets = peak_picking(y,
                          threshold = 0.35,
                          smooth = 7,
                          pre_avg = 0, post_avg = 0,
                          pre_max = 1.0, post_max = 1.0)
    onsets = onsets.astype(np.float) / 100.0
    onsets = combine_events(onsets, 0.03, 'left')
    return np.asarray(onsets)
