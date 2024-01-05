import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal as sig
from matplotlib import pyplot as plt

from utils.signal_utils import DFT, get_fft_freq


# 画出信号
def plot_signal(signal: np.ndarray, title: str = "", save: bool = False, lines: list[float] = []):
    plt.clf()
    plt.cla()
    plt.plot(signal)
    colors = ["r", "g", "b", "y", "m", "c", "k"]
    for index, line in enumerate(lines):
        plt.axvline(x=line, color=colors[index % len(colors)], linestyle="--", linewidth=0.5)
    plt.title(title)
    plt.show()
    title = title.replace(" ", "_")
    if save:
        plt.savefig(title + ".jpg", dpi=300)


# 绘制频谱图
def plot_freq(
    signal: np.ndarray,
    sampling_freq: float,
    r: int,
    title: str = "",
    save: bool = False,
):
    plt.clf()
    plt.cla()
    # f is the frequency array, the zero is in the middle
    N = len(signal)

    signal_fft = DFT(signal, r)

    f = get_fft_freq(N, r, sampling_freq) / 1000

    plt.scatter(f, signal_fft, s=0.1)
    plt.xlabel("Frequency[kHz]")
    plt.ylabel("Amplitude")
    plt.title(title)
    if not save:
        plt.show()
    title = title.replace(" ", "_")
    if save:
        plt.savefig(title + ".jpg", dpi=300)


# 绘制时频图
def plot_spectrogram(
    signal: np.ndarray,
    sampling_freq: float,
    window_size: int,
    title: str = "",
    save: bool = False,
):
    plt.clf()
    plt.cla()
    f, t, Zxx = sig.stft(signal, fs=sampling_freq, nperseg=window_size)
    f /= 1000
    # plot the spectrogram
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=100)
    plt.title(title)
    plt.ylabel("Frequency [kHz]")
    plt.xlabel("Time [sec]")
    if not save:
        plt.show()
    title = title.replace(" ", "_")
    if save:
        plt.savefig(title + ".jpg", dpi=300)
