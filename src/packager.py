from math import ceil
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as sig
from src.modulator import ModulateConfig, Modulator
from utils.plot_utils import plot_signal
from utils.signal_utils import (
    get_frequency_density,
    output_packed_bits,
    output_packed_list,
    read_packed_bits,
)


class FSKPackager:
    def __init__(self, modulator: Modulator):
        self.modulator = modulator
        self.preamble = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        self.len_bits = 16

        assert (
            self.len_bits % self.modulator.config.bits_per_signal == 0
        ), f"len_bits({self.len_bits}) 不是 bits({self.modulator.config.bits_per_signal}) 的整数倍"
        assert (
            len(self.preamble) % self.modulator.config.bits_per_signal == 0
        ), "preamble 长度不是 bits 的整数倍"

    def package(self, data: np.ndarray) -> np.ndarray:
        # 在开头加一个 silent_signal
        silent_signal = np.zeros(self.modulator.config.single_signal_len * 4)

        # 加 preamble，10101010

        # 再用一个长度为 8 的数组来存储 data 的长度
        len_data = len(data)
        assert len_data < 2**self.len_bits, f"数据长度过长, 应该小于 2^{self.len_bits}"
        len_signal = np.array(read_packed_bits(len_data, self.len_bits))

        # 校验码：奇偶校验
        parity = 0 if sum(data) % 2 == 0 else 1
        # 加上 parity

        total_data = np.concatenate(
            [self.preamble, len_signal, data, np.array([parity])]
        )

        print(f"数据包总长度为 {len(total_data)}, 数据负载长度为 {len(data)} ...")

        # 调制
        signal = self.modulator.modulate(total_data)

        # 加上 silent_signal，防止音乐爆鸣
        signal = np.concatenate([silent_signal, signal, silent_signal])

        return signal

    def unpackage(self, signal: np.ndarray) -> np.ndarray:
        single_signal_len = self.modulator.config.single_signal_len

        datapoint_per_signal = 8

        datapoint_span = single_signal_len // datapoint_per_signal
        signal_match = np.zeros(len(signal) // datapoint_span + 1)

        packed_preamble = output_packed_list(
            self.preamble, self.modulator.config.bits_per_signal
        )
        print(packed_preamble)

        for i in range(0, len(signal), datapoint_span):
            local_match = 0
            if i + single_signal_len * len(packed_preamble) > len(signal):
                break
            for j in range(len(packed_preamble)):
                pos_signal = signal[
                    i + j * single_signal_len : i + (j + 1) * single_signal_len
                ]
                local_match += self.modulator.get_power_for_index(
                    pos_signal, packed_preamble[j]
                )
            signal_match[i // datapoint_span] = local_match / len(packed_preamble)

        signal_match = sig.savgol_filter(signal_match, datapoint_per_signal, 1)

        plt.cla()
        plt.clf()
        plt.plot(signal_match)

        # 用相对极值点来找到 preamble 的位置
        signal_start = sig.argrelextrema(
            signal_match, np.greater, order=datapoint_per_signal * len(packed_preamble)
        )[0]

        signal_start = filter(lambda x: signal_match[x] > 0.5, signal_start)
        signal_start = list(signal_start)
        assert len(signal_start) != 0, f"频谱法没有找到 preamble"
        signal_start_value = signal_start[0] * datapoint_span + single_signal_len * len(
            packed_preamble
        )

        # print(signal_start)

        plt.axvline(x=signal_start[0], color="r")
        plt.show()

        # 构建一个 preamble
        modulated_preamble = self.modulator.modulate(self.preamble)

        # 找到第一个 preamble 的位置
        start = 0
        found_start = False

        # 将 signal 与 modulated_preamble 卷积

        # convolved_signal = sig.correlate(modulated_preamble, modulated_preamble, mode="valid")

        convolved_signal = sig.correlate(signal, modulated_preamble, mode="same")

        convolved_signal /= np.max(convolved_signal)

        for i in range(len(convolved_signal)):
            if convolved_signal[i] > 0.9:
                start = i
                found_start = True
                break
        assert found_start, "卷积法没有找到 preamble"

        start += len(modulated_preamble) // 2

        plot_signal(convolved_signal, lines=[start, signal_start_value])

        assert (
            np.abs(start - signal_start_value) <= single_signal_len / 2
        ), f"preamble 的位置不一致"

        # 取得 data_len
        len_data, _ = self.modulator.demodulate(
            signal[start:], self.len_bits // self.modulator.config.bits_per_signal
        )
        print(len_data)

        # 从 len_data 中恢复出 data_len
        dataload_len = output_packed_bits(len_data)

        print(f"数据长度为 {dataload_len}...")

        # 先进行一个滤波
        carrier_freq = self.modulator.config.carrier_freq
        freq_width = self.modulator.config.freq_width
        b, a = sig.iirfilter(
            5,
            carrier_freq / 2,
            btype="highpass",
            fs=self.modulator.config.sampling_freq,
        )

        signal = sig.lfilter(b, a, signal)

        # 取得 data
        start += (
            single_signal_len * self.len_bits // self.modulator.config.bits_per_signal
        )
        data, prob = self.modulator.demodulate(
            signal[start:], ceil(dataload_len / self.modulator.config.bits_per_signal)
        )

        print(f"data: {data.tolist()}")
        # keep 2 decimal places
        print(f"prob: {np.round(prob, 2).tolist()}")

        plt.bar(np.arange(len(prob)), prob)
        plt.axhline(y=0.5, color="r")
        plt.show()

        # 取得 parity
        start = (
            start
            + single_signal_len * dataload_len // self.modulator.config.bits_per_signal
        )
        parity, _ = self.modulator.demodulate(signal[start:], 1)

        # 校验
        # assert sum(data) % 2 == parity, "校验失败"

        return data
