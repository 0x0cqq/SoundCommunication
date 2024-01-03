import numpy as np
from coder.coder import Coder
from modulator.FSK import FSKConfig
from modulator.FSK import FSKModulator
from packager.base import Packager
from utils.plot_utils import plot_signal
from utils.wav_utils import signal_to_wav


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


string_literal = "Hello, World!"

data = coder.encode(string_literal)

signal = packager.package(data)

# plot_signal(signal)

signal_to_wav("output.wav", signal, 48000)
