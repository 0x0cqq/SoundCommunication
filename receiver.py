import numpy as np
from coder.coder import Coder
from modulator.FSK import FSKConfig
from modulator.FSK import FSKModulator
from packager.base import Packager
from utils.plot_utils import plot_signal
from utils.wav_utils import (
    record_to_signal,
    record_to_wav,
    signal_to_wav,
    wav_to_signal,
)


config = FSKConfig(
    sampling_freq=48000,
    amplitude=1,
    signal_duration=0.025,
    carrier_freq=10000,
    freq_shift=2000,
)
modulator = FSKModulator(config)
packager = Packager(None, modulator)
coder = Coder()

record_to_wav("received.wav", 48000, 5)

signal = wav_to_signal("received.wav")

plot_signal(signal)

data = packager.unpackage(signal)

string_literal = coder.decode(data)

print(string_literal)
