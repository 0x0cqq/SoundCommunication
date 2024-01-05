import argparse
from calendar import c

from coder.coder import Coder
from modulator.FSK import FSKConfig
from modulator.FSK import FSKModulator
from packager.base import Packager
from utils.plot_utils import plot_signal
from utils.wav_utils import record_to_wav, signal_to_wav, wav_to_signal


config = FSKConfig(
    sampling_freq=48000,
    amplitude=1,
    signal_duration=0.025,
    carrier_freq=10000,
    freq_shift=100,
)
modulator = FSKModulator(config)
packager = Packager(None, modulator)
coder = Coder()




parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, help="send or receive", default="send", choices=["send", "receive"])

if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == "send":
        string_literal = "Hello, World!"
        data = coder.encode(string_literal)
        signal = packager.package(data)
        # plot_signal(signal[:1000000])
        signal_to_wav("output.wav", signal, 48000)
    elif args.mode == "receive":
        record_to_wav("received.wav", 48000, 5)
        signal = wav_to_signal("received.wav")
        plot_signal(signal)
        data = packager.unpackage(signal)
        string_literal = coder.decode(data)
        print(string_literal)
    else:
        raise NotImplementedError
