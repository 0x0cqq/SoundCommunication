from typing import Callable, Tuple
from utils.code_utils import gray_code, index_to_code

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
        self.freq_shift = np.zeros(2**bits_per_signal, dtype=np.float32)
        gray_codes = gray_code(2, bits_per_signal)
        for i in range(2**bits_per_signal):
            code = gray_codes[i]
            index = int(code, 2)

            self.freq_shift[index] = (
                (i / (2**bits_per_signal - 1) * 2) - 1
            ) * freq_width

        print(f"freq_shift: {self.freq_shift.round(2)}")
        assert (
            freq_width / (2**bits_per_signal - 1) >= 10
        ), "频率分辨率过低, 请增大 freq_width 或者减小 bits_per_signal"
        self.bits_per_signal = bits_per_signal

        self.single_signal_len = int(self.signal_duration * self.sampling_freq)


class Modulator:
    def __init__(self, config: ModulateConfig) -> None:
        self.config = config

    def modulate(self, data: np.ndarray) -> np.ndarray:
        # 首先用 [0] padding 到 bits 对齐
        while len(data) % self.config.bits_per_signal != 0:
            data: np.ndarray = np.concatenate([data, np.array([0])])

        # 生成频率序列
        frequency = []

        for i in range(0, len(data), self.config.bits_per_signal):
            # 编码一次信号
            signal_value = output_packed_bits(data[i : i + self.config.bits_per_signal])
            # 生成信号
            frequency.append(
                np.full(
                    self.config.single_signal_len,
                    self.config.carrier_freq + self.config.freq_shift[signal_value],
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

    def get_power_for_index(self, signal: np.ndarray, index: int, r: int = 10) -> float:
        signal_width = self.config.freq_width * 2 / (2**self.config.bits_per_signal)
        return get_frequency_density(
            signal,
            self.config.sampling_freq,
            r,
            self.config.carrier_freq + self.config.freq_shift[index],
            signal_width / 2,
        )

    def demodulate(
        self, signal: np.ndarray, signal_num: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 输入的 signal 是已经去掉了前导和后导的
        # 单个信号的长度
        signal_len = self.config.single_signal_len
        # 信号的个数
        signal_num = int(signal_num)
        if signal_num == -1:
            signal_num = round(len(signal) / signal_len)
        data = []
        probs = []
        for i in range(signal_num):
            start, end = i * signal_len, (i + 1) * signal_len
            this_signal = signal[start:end]
            # 取到最大功率的频率
            most_likely_freq = get_most_likely_freq(
                this_signal, self.config.sampling_freq, 16
            )
            max_freq_from_most_likely_freq = np.argmin(
                np.abs(
                    np.array(self.config.freq_shift)
                    + self.config.carrier_freq
                    - most_likely_freq
                )
            )
            # print(f"most likely freq: {most_likely_freq}, error: {np.abs(self.config.freq_shift[max_freq_from_most_likely_freq] + self.config.carrier_freq - most_likely_freq)}")

            # 计算和哪个信号最接近
            powers = [
                self.get_power_for_index(this_signal, j, 16)
                for j in range(2**self.config.bits_per_signal)
            ]
            # print(f"powers for {i}: {[round(i, 2) for i in powers]}")
            max_freq = np.argmax(powers)

            if max_freq != max_freq_from_most_likely_freq:
                print("信号解调不一致，很可能错误")
                max_freq_error = np.abs(
                    self.config.freq_shift[max_freq_from_most_likely_freq]
                    + self.config.carrier_freq
                    - most_likely_freq
                )
                if (
                    max_freq_error
                    < self.config.freq_width / (2**self.config.bits_per_signal) * 0.66
                    and powers[max_freq] <= 0.4
                ):
                    max_freq = max_freq_from_most_likely_freq
                    print(
                        f"使用频率最大似然解码, error: {max_freq_error}, power: {powers[max_freq]}"
                    )
                else:
                    print(
                        f"使用最大功率的频率区间解码, error: {max_freq_error}, power: {powers[max_freq]}"
                    )

            # 获得概率
            prob = powers / np.sum(powers)

            # 解到二进制
            this_data = read_packed_bits(max_freq, self.config.bits_per_signal)
            data.extend(this_data)

            # 记录概率
            probs.append(prob[max_freq])

        return np.array(data), np.array(probs)
