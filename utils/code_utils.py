# From https://github.com/JHaller27/GrayCodeGenerator


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
