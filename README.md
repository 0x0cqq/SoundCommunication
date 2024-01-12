# Sound Communication

2023-2024 学年 秋季学期 物联网原理 大作业

## 程序环境配置

程序运行在 Python 环境中，测试环境为 `python 3.10` .

1. 创建虚拟环境：`conda create -n iot python=3.10`
2. 激活虚拟环境：`conda activate iot`
3. 安装依赖：`pip install -r requirements.txt`

## 程序运行

### 发送信号

1. 将要发送的字符串在程序 `main.py` 中编码
2. 运行 `python main.py --mode send`

程序将会生成 `output.wav` 放在根目录下。

### 接收信号

> 这里为了方便确定录音时长，在接收信号的时候也需要输入字符串，但我们只会利用字符串的长度，而且字符串的长度除了录音长度不会在任何解码中使用。

1. 将要接收的字符串在程序 `main.py` 中编码
2. 运行 `python main.py --mode receive`

程序会将录音文件存储到 `received.wav` 下，并将解码的结果通过命令行输出的形式输出。

## 文档

见 [Doc](doc.md) 。
