from typing import Callable, Tuple

from utils.plot_utils import plot_freq, plot_signal
from utils.signal_utils import get_most_likely_freq
from modulator.base import BaseConfig, BaseModulator
import numpy as np
import scipy.fft as fft
import scipy.signal as sig
import scipy.ndimage as ndimage


class FSKConfig(BaseConfig):
    def __init__(
        self,
        sampling_freq: float,
        amplitude: float,
        signal_duration: float,
        carrier_freq: float,
        freq_shift: float,
    ) -> None:
        super().__init__(sampling_freq, amplitude, signal_duration)
        # 这里是载波频率
        self.carrier_freq = carrier_freq
        # 这里是频移的量
        self.freq_shift = freq_shift

class FSKModulator(BaseModulator):
    def __init__(self, config: FSKConfig) -> None:
        self.config = config

    def modulate(self, data: np.ndarray) -> np.ndarray:
        # 时间
        T = np.arange(0, self.config.signal_duration, 1 / self.config.sampling_freq)
        signal_0 = np.full_like(T, self.config.carrier_freq - self.config.freq_shift)
        signal_1 = np.full_like(T, self.config.carrier_freq + self.config.freq_shift)
        signal = np.concatenate([signal_0 if data[i] == 0 else signal_1 for i in range(len(data))])

        # Gaussian filter

        frequency = ndimage.gaussian_filter(signal, 4)

        plot_signal(signal, "frequency after gaussian filter", True)

        # caculate phase
        phase = np.cumsum(2 * np.pi * frequency / self.config.sampling_freq)

        # caculate signal
        signal = self.config.amplitude * np.sin(phase)

        return signal

    def demodulate(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 输入的 signal 是已经去掉了前导和后导的
        # 单个信号的长度
        T = np.arange(0, self.config.signal_duration, 1 / self.config.sampling_freq)
        signal_0 = np.sin(
            2 * np.pi * (self.config.carrier_freq - self.config.freq_shift) * T
        )
        signal_1 = np.sin(
            2 * np.pi * (self.config.carrier_freq + self.config.freq_shift) * T
        )
        signal_len = len(signal_0)
        # 信号的个数
        signal_num = round(len(signal) / signal_len)
        data = []
        prob = []
        for i in range(signal_num):
            start, end = i * signal_len, (i + 1) * signal_len
            this_signal = signal[start:end]
            # 对 this_signal 做 FFT
            most_likely_freq = get_most_likely_freq(this_signal, self.config.sampling_freq, 10)
            prob_0 = np.abs(most_likely_freq - (self.config.carrier_freq - self.config.freq_shift))
            prob_1 = np.abs(most_likely_freq - (self.config.carrier_freq + self.config.freq_shift))
            if prob_0 < prob_1:
                data.append(0)
                prob.append((prob_0) / (prob_0 + prob_1))
            else:
                data.append(1)
                prob.append((prob_1) / (prob_0 + prob_1))
        return np.array(data), np.array(prob)
