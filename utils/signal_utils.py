from typing import Callable
import numpy as np
import scipy.fft as fft
import scipy.signal as sig

from matplotlib import pyplot as plt
# from utils.plot_utils import plot_signal


def DFT(x: np.ndarray, r: int = 1) -> np.ndarray:
    fft_x = fft.fft(x, r * len(x))
    # 将负频率放到左侧
    fft_x = np.abs(np.fft.fftshift(fft_x))
    return fft_x

def DFT_N(N: int, f: Callable[[int], np.ndarray], r: int = 1):
    x = f(N)
    fft_x = DFT(x, r)
    return fft_x

def get_fft_freq(N: int, r: int, sampling_freq: float):
    return np.linspace(-r * N / 2, r * N / 2 - 1, r * N) * sampling_freq / (r * N)

def get_most_likely_freq(signal: np.ndarray, sampling_freq: float, r: int) -> float:
    N = len(signal)
    signal_fft = DFT(signal, r)
    freq = get_fft_freq(N, r, sampling_freq)
    # add the absolute value of the negative frequency to the positive frequency
    N_fft = len(signal_fft)
    signal_fft += signal_fft[::-1]
    # remove the negative frequency at the left
    signal_fft = signal_fft[N_fft // 2 :]
    freq = freq[N_fft // 2 :]
    # plt.cla()
    # plt.clf()
    # plt.plot(freq, signal_fft)
    # plt.show()
    # do a smooth
    signal_fft = sig.savgol_filter(signal_fft, 5, 2)
    # plt.cla()
    # plt.clf()
    # plt.plot(freq, signal_fft)
    max_index = np.argmax(signal_fft)
    # plt.axvline(x=freq[max_index], color="r")
    # plt.show()
    # find the most likely frequency
    return freq[max_index]

def get_frequency_density(signal: np.ndarray, sampling_freq: float, r: int, freq: float) -> float:
    N = len(signal)
    signal_fft = DFT(signal, r)
    freq_array = get_fft_freq(N, r, sampling_freq)
    # add the absolute value of the negative frequency to the positive frequency
    N_fft = len(signal_fft)
    signal_fft += signal_fft[::-1]
    # remove the negative frequency at the left
    signal_fft = signal_fft[N_fft // 2 :]
    freq_array = freq_array[N_fft // 2 :]
    # do a smooth
    signal_fft = sig.savgol_filter(signal_fft, 5, 2)
    # find the most likely frequency
    return signal_fft[np.argmin(np.abs(freq_array - freq))]

