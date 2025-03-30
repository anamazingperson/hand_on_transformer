import torch
import torch.nn as nn
from config import Config
from models import Transformer
from utils import generate_data, create_mask


model = Transformer(Config.vocab_size).to(Config.device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)

for epoch in range(Config.epochs):
    src, tgt = generate_data(Config.batch_size, Config.max_len, Config.vocab_size)
    src_mask, tgt_mask = create_mask(src, tgt)
    
    output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
    loss = criterion(output.view(-1, Config.vocab_size), tgt[:, 1:].contiguous().view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 测试模型
with torch.no_grad():
    test_src, test_tgt = generate_data(1, 5, Config.vocab_size)
    test_src_mask, test_tgt_mask = create_mask(test_src, test_tgt)
    output = model(test_src, test_tgt[:, :-1], test_src_mask, test_tgt_mask[:, :, :-1, :-1])
    predicted = output.argmax(-1)
    print("\nTest Example:")
    print("Input:  ", test_src[0].cpu().numpy())
    print("Output: ", predicted[0].cpu().numpy())