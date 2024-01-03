import numpy as np


class BaseConfig:
    def __init__(
        self, sampling_freq: float, amplitude: float, signal_duration: float
    ) -> None:
        self.sampling_freq = sampling_freq
        self.amplitude = amplitude
        self.signal_duration = signal_duration


class BaseModulator:
    def __init__(self, **kwargs):
        """初始化调制器"""
        pass

    def modulate(self, data: np.ndarray) -> np.ndarray:
        """调制信号

        Args:
            data (np.ndarray): 待调制的信号，0/1 数组

        Raises:
            NotImplementedError: 虚基类

        Returns:
            np.ndarray: 调制好的信号
        """
        raise NotImplementedError

    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """解调信号

        Args:
            signal (np.ndarray): 待解调的信号，实数数组
            kwargs: 其他解调参数

        Raises:
            NotImplementedError: 虚基类

        Returns:
            np.ndarray: 解调好的信号，0/1 数组
        """
        raise NotImplementedError
