import pyaudio
import scipy.io.wavfile as wav
import numpy as np


# 写入 WAV 文件
def signal_to_wav(filename: str, data: np.ndarray, sampling_freq: float):
    wav.write(filename, int(sampling_freq), np.int16(data * 32767.0))


# 从 WAV 文件读取信号
def wav_to_signal(filename: str) -> np.ndarray:
    return wav.read(filename)[1] / 32767.0


# 从麦克风录制到信号
def record_to_signal(sampling_freq: float, time: float) -> np.ndarray:
    # record component
    p = pyaudio.PyAudio()
    # open the stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=int(sampling_freq),
        input=True,
        frames_per_buffer=1024,
    )
    # read the data by frames
    frames = []
    print(f"Recording for {time.__round__(2)} second...", flush=True)
    for _ in range(0, int(sampling_freq / 1024 * time)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("End recording.", flush=True)
    # join the frames
    return np.frombuffer(b"".join(frames), dtype=np.int16)


# 从麦克风录制到 WAV 文件
def record_to_wav(filename: str, sampling_freq: float, time: float):
    data = record_to_signal(sampling_freq, time)
    data = data / 32767.0
    signal_to_wav(filename, data, sampling_freq)
