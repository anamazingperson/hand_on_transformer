import torch
import torch.nn as nn
import math


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, device, dropout=0.1):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model,device=device)  # 词嵌入
        self.pos_emb = PositionalEncoding(d_model, max_len, device)  # 位置编码
        self.dropout = nn.Dropout(dropout)  # Dropout 层

    def forward(self, x):
        tok_emb = self.tok_emb(x)  # 获取词嵌入
        pos_emb = self.pos_emb(x)  # 获取位置编码
        return self.dropout(tok_emb + pos_emb)  # 将词嵌入和位置编码相加，并应用 Dropout
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model,max_len,device):
        super(PositionalEncoding,self).__init__()
        self.position = torch.zeros(max_len,d_model,device=device)
        self.position.requires_grad = False
        pos = torch.arange(0,max_len,device=device).unsqueeze(1)
        _2i = torch.arange(0,d_model,step=2,device=device)
        self.position[:,0::2] = torch.sin(pos/1000**(_2i/d_model))
        self.position[:,1::2] = torch.cos(pos/1000**(_2i/d_model))
    def forward(self,x):
        batch_size,length = x.size()
        return self.position[:length,:]


class Encoder(nn.Module):
    def __init__(self,d_model,dk,dv,hidd1,device):
        super().__init__()
        self.dk = dk
        self.head = d_model//dv
        self.d_model = d_model
        self.dv = dv
        self.device = device
        self.fc1 = nn.Linear(d_model,hidd1,device=device)
        self.fc2 = nn.Linear(hidd1,d_model,device=device)
        self.norm = nn.LayerNorm((d_model),device=device)
        self.Wq = nn.Parameter(torch.rand(self.d_model,self.head*self.dk,device=self.device))
        self.Wk = nn.Parameter(torch.rand(self.d_model,self.head*self.dk,device=self.device))
        self.Wv = nn.Parameter(torch.rand(self.d_model,self.head*self.dv,device=self.device))
    def Mutilattention(self,X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        Q = Q.view(-1,Q.shape[1],self.head,self.dk).transpose(1,2)
        K = K.view(-1,K.shape[1],self.head,self.dk).transpose(1,2)
        V = V.view(-1,V.shape[1],self.head,self.dv).transpose(1,2)
        score = Q @ K.transpose(-1,-2)/math.sqrt(self.dk) 
        weigth = torch.softmax(score,dim=-1) @ V
        return weigth.transpose(1,2).contiguous().view(-1,weigth.shape[2],self.head*self.dv)

    def forward(self,X):
        # 多头怎么实现
        AttentionOut = self.Mutilattention(X)
        AttentionOut = self.norm(X+AttentionOut)
        out = self.fc2(torch.relu(self.fc1(AttentionOut)))
        return out



class Decoder(nn.Module):
    def __init__(self,d_model,dk,dv,hidd,device):
        super(Decoder,self).__init__()
        self.head = d_model//dv
        self.WqM = nn.Parameter(torch.rand(d_model,self.head*dk,device=device))
        self.WkM = nn.Parameter(torch.rand(d_model,self.head*dk,device=device))
        self.WvM = nn.Parameter(torch.rand(d_model,self.head*dv,device=device))
        self.Wq = nn.Parameter(torch.rand(d_model,self.head*dk,device=device))
        self.Wk = nn.Parameter(torch.rand(d_model,self.head*dk,device=device))
        self.Wv = nn.Parameter(torch.rand(d_model,self.head*dv,device=device))
        self.dk = dk
        self.dv = dv
        self.device = device
        self.norm = nn.LayerNorm(d_model,device=device)
        self.fc1 = nn.Linear(d_model,hidd,device=device)
        self.fc2 = nn.Linear(hidd,d_model,device=device)
    def MaskMultiHead(self,X):

        Q = X @ self.WqM
        K = X @ self.WkM
        V = X @ self.WvM
        Q = Q.view(-1,Q.shape[1],self.head,self.dk).transpose(1,2)
        K = K.view(-1,K.shape[1],self.head,self.dk).transpose(1,2)
        V = V.view(-1,V.shape[1],self.head,self.dv).transpose(1,2)
        Mask = torch.tril(torch.ones(X.shape[1],X.shape[1],device=self.device))
        Mask = Mask.unsqueeze(0).unsqueeze(0)
        score = Q @ K.transpose(-1,-2)/math.sqrt(self.dk) * Mask

        weigth = torch.softmax(score,dim=-1)@ V
        return weigth.transpose(1,2).contiguous().view(-1,weigth.shape[2],self.head*self.dv)
    def MultiHead(self,X,X_encoder):
        Q = X @ self.WqM
        K = X_encoder @ self.WkM
        V = X_encoder @ self.WvM
        Q = Q.view(-1,Q.shape[1],self.head,self.dk).transpose(1,2)
        K = K.view(-1,K.shape[1],self.head,self.dk).transpose(1,2)
        V = V.view(-1,V.shape[1],self.head,self.dv).transpose(1,2)
        score = Q @ K.transpose(-1,-2)/math.sqrt(self.dk)

        weigth = torch.softmax(score,dim=-1)@ V
        return weigth.transpose(1,2).contiguous().view(-1,weigth.shape[2],self.head*self.dv)

    def forward(self,x):
        x,X_encoder = x
        fx = self.MaskMultiHead(x)
        fx = self.MultiHead(fx,X_encoder)
        x = self.norm(x+fx)
        return self.fc2(torch.relu(self.fc1(x)))
    

class Transformer(nn.Module):
    def __init__(self,vocab_size,max_len,d_model,dk,dv,hidd,out_dim,device):
        super(Transformer,self).__init__()
        self.embedding = TransformerEmbedding(vocab_size,d_model,max_len,device)
        self.encoder = Encoder(d_model,dk,dv,hidd,device)
        self.decoder = Decoder(d_model,dk,dv,hidd,device)
        self.fc = nn.Linear(d_model,out_dim,device=device)

    def forward(self,x):
        x = self.embedding(x)
        encoder_x = self.encoder(x)
        x = self.decoder((x,encoder_x))
        x = self.fc(x)
        return torch.softmax(x,dim=-1)
vocab_size = 10000
d_model = 512
max_len = 512
dk = 4
dv = 8
hidd = 10
out_dim = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding = TransformerEmbedding(vocab_size,d_model,max_len,device)

sequence = torch.randint(0,20,size=(10,10),device=device)


trans = Transformer(vocab_size,max_len,d_model,dk,dv,hidd,out_dim,device)


def get_train_data(length):
    x = torch.arange(0,7,step=0.01)
    y = torch.sin(x)
    return x,y


x,y = get_train_data(100)
x = x.to(device)
y = y.to(device)
x = x.view(-1,1)
y  = y.view(-1,1)
epoch = 100
batch_size = 50
optim = torch.optim.Adam(trans.parameters(),lr = 0.01)
cretira = nn.MSELoss()
for i in range(epoch):
    y_pre = trans(x)
    loss = cretira(y_pre,x)
    optim.zero_grad()
    cretira.backward()
    optim.step()
    if epoch%10 ==0:
        print(loss)
