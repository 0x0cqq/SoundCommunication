from .base import BaseConfig, BaseModulator
import numpy as np

class FSKConfig(BaseConfig):
    def __init__(self, sampling_freq: float, amplitude: float,  signal_duration: float, carrier_freq: float, freq_shift: float) -> None:
        super().__init__(sampling_freq, amplitude, signal_duration)
        # 这里是载波频率
        self.carrier_freq = carrier_freq
        # 这里是频移的量
        self.freq_shift = freq_shift
            

class FSKModulator(BaseModulator):
    def __init__(self, config: FSKConfig) -> None:
        self.config = config
    
    def modulate(self, data: np.ndarray) -> np.ndarray:
        # 调制的时候，不考虑前面的 premble
        T = np.arange(0, self.config.signal_duration, 1 / self.config.sampling_freq)
        signal_0 = np.sin(2 * np.pi * (self.config.carrier_freq - self.config.freq_shift) * T)
        signal_1 = np.sin(2 * np.pi * (self.config.carrier_freq + self.config.freq_shift) * T)
        signal = np.concatenate([signal_0 if bit == 0 else signal_1 for bit in data])
        return signal
    
    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        # 输入的 signal 是已经去掉了前导和后导的
        # 单个信号的长度
        T = np.arange(0, self.config.signal_duration, 1 / self.config.sampling_freq)
        signal_0 = np.sin(2 * np.pi * (self.config.carrier_freq - self.config.freq_shift) * T)
        signal_1 = np.sin(2 * np.pi * (self.config.carrier_freq + self.config.freq_shift) * T)
        signal_len = len(signal_0)
        # 信号的个数
        signal_num = round(len(signal) / signal_len)
        data = []
        for i in range(signal_num):
            start, end = i * signal_len, (i + 1) * signal_len
            this_signal = signal[start:end]
            # 点积
            dot_0 = abs(np.dot(this_signal, signal_0[:len(this_signal)]))
            dot_1 = abs(np.dot(signal[start:end], signal_1[:len(this_signal)]))
            data.append(0 if dot_0 > dot_1 else 1)
        return np.array(data)


