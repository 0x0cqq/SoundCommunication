import numpy as np


class Coder:
    def __init__(self):
        pass

    def encode(self, str: str) -> np.ndarray:
        # use ascii
        data = []
        for char in str:
            num = ord(char)
            for i in range(8):
                data.append(num % 2)
                num //= 2
        return np.array(data)

    def decode(self, data: np.ndarray) -> str:
        assert len(data) % 8 == 0, "数据长度不是 8 的倍数"
        chars = []
        for i in range(len(data) // 8):
            num = 0
            for j in range(8):
                num += data[i * 8 + j] * (2**j)
            chars.append(chr(num))
        return "".join(chars)
