from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as sig
from modulator.base import BaseModulator
from utils.plot_utils import plot_signal


class PackageConfig:
    def __init__(
        self,
        sampling_freq: float,
        amplitude: float,
        signal_duration: float,
    ):
        self.sampling_freq = sampling_freq
        self.amplitude = amplitude
        self.signal_duration = signal_duration


class Packager:
    def __init__(self, config: PackageConfig, modulator: BaseModulator):
        # self.config = config
        self.modulator = modulator
        self.preamble = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    def package(self, data: np.ndarray) -> np.ndarray:
        # 在开头加一个 silent_signal
        silent_signal = np.zeros_like(self.modulator.modulate(np.array([1, 0])))

        # 加 preamble，10101010
        

        # 再用一个长度为 8 的数组来存储 data 的长度
        len_data = len(data)
        assert len_data < 2**8, "数据长度过长, 应该小于 2^8"
        len_signal = np.zeros(8)
        for i in range(8):
            len_signal[i] = len_data % 2
            len_data //= 2

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
        # 先进行一个带通滤波
        b, a = sig.iirfilter(
            1, 1000, btype="highpass", fs=self.modulator.config.sampling_freq
        )

        signal = sig.lfilter(b, a, signal)

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
            if convolved_signal[i] > 0.85:
                start = i
                found_start = True
                break
        assert found_start, "没有找到 preamble"


        plot_signal(convolved_signal, line=start)

        start += len(modulated_preamble) // 2

        # 取得 data_len
        single_signal_len = len(self.modulator.modulate(np.array([1])))
        len_data, _ = self.modulator.demodulate(
            signal[start : start + single_signal_len * 8]
        )
        print(len_data)

        # 从 len_data 中恢复出 data_len
        data_len = 0
        for i in range(8):
            data_len += len_data[i] * (2**i)

        print(f"数据长度为 {data_len}...")

        # 取得 data
        start = start + single_signal_len * 8
        data, prob = self.modulator.demodulate(
            signal[start : start + single_signal_len * data_len]
        )

        print(f"data: {data.tolist()}")
        # keep 2 decimal places
        print(f"prob: {np.round(prob, 2).tolist()}")

        plt.bar(np.arange(len(prob)), prob)
        plt.axhline(y=0.5, color="r")
        plt.show()

        # 取得 parity
        start = start + single_signal_len * data_len
        parity, _ = self.modulator.demodulate(signal[start : start + single_signal_len])

        # 校验
        # assert sum(data) % 2 == parity, "校验失败"

        return data
