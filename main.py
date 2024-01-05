import argparse
from calendar import c

from src.coder import Coder
from src.modulator import ModulateConfig
from src.modulator import Modulator
from src.packager import FSKPackager
from utils.plot_utils import plot_signal
from utils.wav_utils import record_to_wav, signal_to_wav, wav_to_signal


config = ModulateConfig(
    sampling_freq=48000,
    amplitude=1,
    signal_duration=0.05,
    carrier_freq=10000,
    freq_width=2000,
    bits_per_signal=4,
)
modulator = Modulator(config)
packager = FSKPackager(modulator)
coder = Coder()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    help="send or receive",
    default="send",
    choices=["send", "receive"],
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == "send":
        string_literal = "Hello, World! This is a test from FSK! Glad to hear that you can receive this message! Hopefully you can decode this message!"
        data = coder.encode(string_literal)
        signal = packager.package(data)
        signal_to_wav("output.wav", signal, 48000)
    elif args.mode == "receive":
        record_to_wav("received.wav", 48000, 20)
        signal = wav_to_signal("received.wav")
        plot_signal(signal)
        data = packager.unpackage(signal)
        string_literal = coder.decode(data)
        print(string_literal)
    else:
        raise NotImplementedError
