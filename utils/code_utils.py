# From https://github.com/JHaller27/GrayCodeGenerator


import numpy as np


def replace(s: str, c: str, i: int) -> str:
    return "".join([s[x] if x != i else c for x in range(len(s))])


def inc(d: str, base: int) -> str:
    D = (int(d, base) + 1) % base
    return str(D) if D < 10 else chr((D - 10) + ord("A"))


def gray_code(base: int, digits: int):
    num = "0" * digits
    seq = [num]

    while len(seq) < base**digits:
        idx = len(num) - 1
        while replace(num, inc(num[idx], base), idx) in seq:
            idx -= 1
        num = replace(num, inc(num[idx], base), idx)
        seq.append(num)

    return seq


def code_to_index(code: str, base: int):
    return int(code, base)


def index_to_code(index: int, base: int, digits: int):
    return gray_code(base, digits)[index]


def get_parity(data: np.ndarray) -> int:
    return int(np.sum(data)) % 2

def generate_hamming_code(data: np.ndarray):
    length = len(data)
    parity_bits = int(np.ceil(np.log2(length)))
    
    # 生成最终结果
    res = np.zeros(length + parity_bits, dtype=np.int32)

    # 填入数据
    current_pos = 1
    for i in range(len(res)):
        if i + 1 in [2**x for x in range(parity_bits)]:
            continue
        res[i] = data[current_pos - 1]
        current_pos += 1
    
    # 填入校验位
    for i in range(parity_bits):
        pos = 2 ** i
        for j in range(len(res)):
            if (j + 1) & pos != 0 and j + 1 != pos:
                res[pos - 1] ^= res[j]

    return res

def decode_hamming_code(data: np.ndarray) -> np.ndarray:
    partial_bits = int(np.ceil(np.log2(len(data))))
    for i in range(len(data)):
        # 跳过校验位
        if i + 1 in [2**x for x in range(partial_bits)]:
            continue
        # 计算校验位
        parity = 0
        for j in range(partial_bits):
            if (j + 1) & (i + 1) != 0 and j + 1 != i + 1:
                parity ^= data[j]
        if parity != data[i]:
            print(f"校验位 {i + 1} 错误")
    
    # 去掉校验位
    res = []
    for i in range(len(data)):
        if i + 1 not in [2**x for x in range(partial_bits)]:
            res.append(data[i])
    return np.array(res)


