import torch
from config import Config

def generate_data(batch_size, seq_len, vocab_size):
    # 生成简单的复制任务数据
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt = src.clone()
    return src.to(Config.device), tgt.to(Config.device)

def create_mask(src, tgt):
    src_mask = torch.ones_like(src).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
    tgt_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1))).unsqueeze(0).unsqueeze(0)
    return src_mask.to(Config.device), tgt_mask.to(Config.device)