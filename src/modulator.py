from typing import Callable, Tuple

from utils.signal_utils import (
    get_frequency_density,
    get_most_likely_freq,
    output_packed_bits,
    read_packed_bits,
)
import numpy as np
import scipy.ndimage as ndimage


class ModulateConfig:
    def __init__(
        self,
        sampling_freq: float,
        amplitude: float,
        signal_duration: float,
        carrier_freq: float,
        freq_width: float = 100,
        bits_per_signal: int = 1,
    ) -> None:
        self.sampling_freq = sampling_freq
        self.amplitude = amplitude
        self.signal_duration = signal_duration
        # 这里是载波频率
        self.carrier_freq = carrier_freq
        self.freq_width = freq_width  # 这个宽度是单边的
        # 这里是频移的量，均分到每个 bit 上
        self.freq_shift = []
        for i in range(2**bits_per_signal):
            self.freq_shift.append(
                (i + 1) * (2 * freq_width) / (2**bits_per_signal) - freq_width
            )
        self.bits_per_signal = bits_per_signal


class Modulator:
    def __init__(self, config: ModulateConfig) -> None:
        self.config = config

    def modulate(self, data: np.ndarray) -> np.ndarray:
        # 首先用 [0] padding 到 bits 对齐
        while len(data) % self.config.bits_per_signal != 0:
            data: np.ndarray = np.concatenate([data, np.array([0])])
        # 时间
        T = np.arange(0, self.config.signal_duration, 1 / self.config.sampling_freq)

        # 生成频率序列
        frequency = []

        for i in range(0, len(data), self.config.bits_per_signal):
            # 编码一次信号
            signal_value = output_packed_bits(data[i : i + self.config.bits_per_signal])
            # 生成信号
            frequency.append(
                np.full_like(
                    T, self.config.carrier_freq + self.config.freq_shift[signal_value]
                )
            )

        frequency = np.concatenate(frequency)

        # 对频率序列做高斯滤波
        frequency = ndimage.gaussian_filter(frequency, 4)

        # 计算相位
        phase = np.cumsum(2 * np.pi * frequency / self.config.sampling_freq)

        # 根据相位生成信号
        signal = self.config.amplitude * np.sin(phase)

        return signal

    def demodulate(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 输入的 signal 是已经去掉了前导和后导的
        # 单个信号的长度
        T = np.arange(0, self.config.signal_duration, 1 / self.config.sampling_freq)
        signal_len = len(T)
        # 信号的个数
        signal_num = round(len(signal) / signal_len)
        data = []
        probs = []
        for i in range(signal_num):
            start, end = i * signal_len, (i + 1) * signal_len
            this_signal = signal[start:end]
            # 取到最大功率的频率
            most_likely_freq = get_most_likely_freq(
                this_signal, self.config.sampling_freq, 10
            )
            # 计算和哪个信号最接近
            signal_width = (
                self.config.freq_width * 2 / (2**self.config.bits_per_signal)
            )
            powers = [
                get_frequency_density(
                    this_signal,
                    self.config.sampling_freq,
                    10,
                    self.config.carrier_freq + self.config.freq_shift[j],
                    signal_width / 2,
                )
                for j in range(2**self.config.bits_per_signal)
            ]
            print(f"powers for {i}: {powers}")
            max_freq = np.argmax(powers)
            # softmax 获得概率
            prob = np.exp(powers) / np.sum(np.exp(powers))

            # 解到二进制
            this_data = read_packed_bits(max_freq, self.config.bits_per_signal)
            data.extend(this_data)

            # 记录概率
            probs.append(prob[max_freq])

        return np.array(data), np.array(probs)
