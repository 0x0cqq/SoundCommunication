from typing import Callable
import numpy as np
import scipy.fft as fft
import scipy.signal as sig

from matplotlib import pyplot as plt
# from utils.plot_utils import plot_signal


def DFT(x: np.ndarray, r: int = 1) -> np.ndarray:
    fft_x = fft.fft(x, r * len(x), norm="forward")
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
    # 进行一个 DFT
    signal_fft = DFT(signal, r)
    freq = get_fft_freq(N, r, sampling_freq)
    # 把负频率的部分加到正频率上
    N_fft = len(signal_fft)
    signal_fft += signal_fft[::-1]
    signal_fft = signal_fft[N_fft // 2 :]
    freq = freq[N_fft // 2 :]
    # 进行一个 smooth
    signal_fft = sig.savgol_filter(signal_fft, 5, 2)
    # 找到最大值
    max_index = np.argmax(signal_fft)
    return freq[max_index]

def get_frequency_density(signal: np.ndarray, sampling_freq: float, r: int, freq: float, width: float) -> float:
    N = len(signal)
    signal_fft = DFT(signal, r)
    freq_array = get_fft_freq(N, r, sampling_freq)
    N_fft = len(signal_fft)
    signal_fft += signal_fft[::-1]
    signal_fft = signal_fft[N_fft // 2 :]
    freq_array = freq_array[N_fft // 2 :]
    # do a smooth
    signal_fft = sig.savgol_filter(signal_fft, 5, 2)
    # find the most likely frequency
    index_low = np.argmin(np.abs(freq_array - (freq - width)))
    index_high = np.argmin(np.abs(freq_array - (freq + width)))
    # print(f"index_low: {index_low}, index_high: {index_high}")
    # 获取该频段的能量占比，功率和幅度平方成正比
    return np.sum(signal_fft[index_low:index_high] ** 2) / np.sum(signal_fft ** 2)

# 端序：小端序
def read_packed_bits(num: int, bits: int) -> list:
    res = []
    for i in range(bits):
        res.append(num % 2)
        num //= 2
    return res[::-1]

# 端序：小端序
def output_packed_bits(data: list) -> int:
    res = 0
    for i in range(len(data)):
        res = res * 2 + data[i]
    return res