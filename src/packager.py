from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as sig
from src.modulator import ModulateConfig, Modulator
from utils.plot_utils import plot_signal
from utils.signal_utils import get_frequency_density, output_packed_bits, read_packed_bits


class FSKPackager:
    def __init__(self, modulator: Modulator):
        self.modulator = modulator
        self.preamble = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.single_signal_len = int(
            self.modulator.config.signal_duration * self.modulator.config.sampling_freq
        )

    def package(self, data: np.ndarray) -> np.ndarray:
        # 在开头加一个 silent_signal
        silent_signal = np.zeros(self.single_signal_len * 4)

        # 加 preamble，10101010

        # 再用一个长度为 8 的数组来存储 data 的长度
        len_data = len(data)
        assert len_data < 2**8, "数据长度过长, 应该小于 2^8"
        len_signal = np.array(read_packed_bits(len_data, 8))

        # 校验码：奇偶校验
        parity = 0 if sum(data) % 2 == 0 else 1
        # 加上 parity

        new_data = np.concatenate([self.preamble, len_signal, data, np.array([parity])])

        print(f"数据包总长度为 {len(new_data)}, 数据长度为 {len(data)} ...")

        # 调制
        signal = self.modulator.modulate(new_data)

        # 加上 silent_signal，防止音乐爆鸣
        signal = np.concatenate([silent_signal, signal, silent_signal])

        return signal

    def unpackage(self, signal: np.ndarray) -> np.ndarray:
        # 先进行一个滤波
        carrier_freq = self.modulator.config.carrier_freq
        freq_width = self.modulator.config.freq_width
        b, a = sig.iirfilter(
            3,
            [carrier_freq - freq_width * 2, carrier_freq + freq_width * 2],
            btype="bandpass",
            fs=self.modulator.config.sampling_freq,
        )

        signal = sig.lfilter(b, a, signal)

        datapoint_span = self.single_signal_len // 8
        signal_total_density = np.zeros(len(signal) // datapoint_span + 1)

        for i in range(0, len(signal), datapoint_span):
            signal_total_density[i // datapoint_span] = get_frequency_density(
                signal[i : i + self.single_signal_len],
                self.modulator.config.sampling_freq,
                10,
                self.modulator.config.carrier_freq,
                self.modulator.config.freq_width,
            )

        plt.cla()
        plt.clf()
        plt.plot(signal_total_density)
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
        assert found_start, "没有找到 preamble"

        plot_signal(convolved_signal, line=start)

        start += len(modulated_preamble) // 2

        # 取得 data_len
        len_data, _ = self.modulator.demodulate(
            signal[start : start + self.single_signal_len * 8]
        )
        print(len_data)

        # 从 len_data 中恢复出 data_len
        data_len = output_packed_bits(len_data)

        print(f"数据长度为 {data_len}...")

        # 取得 data
        start = start + self.single_signal_len * 8
        data, prob = self.modulator.demodulate(
            signal[start : start + self.single_signal_len * data_len]
        )

        print(f"data: {data.tolist()}")
        # keep 2 decimal places
        print(f"prob: {np.round(prob, 2).tolist()}")

        plt.bar(np.arange(len(prob)), prob)
        plt.axhline(y=0.5, color="r")
        plt.show()

        # 取得 parity
        start = start + self.single_signal_len * data_len
        parity, _ = self.modulator.demodulate(
            signal[start : start + self.single_signal_len]
        )

        # 校验
        # assert sum(data) % 2 == parity, "校验失败"

        return data
