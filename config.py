import torch
class Config:
    vocab_size = 1000   # 词汇表大小
    max_len = 64        # 序列最大长度
    d_model = 128       # 模型维度
    num_heads = 8       # 多头注意力头数
    d_ff = 512          # 前馈网络隐藏层维度
    num_layers = 2      # 编码器/解码器层数
    dropout = 0.1       # Dropout 概率
    batch_size = 32     # 批大小
    epochs = 10         # 训练轮数
    lr = 0.001          # 学习率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()