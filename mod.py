import torch.nn as nn
import torch
EMB_DIM = 64
HID_DIM = 32
NUM_LAYERS = 2
class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, h_dim, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, h_dim, num_layers=num_layers, dropout=dropout)
        self.num_layers = num_layers
        self.h_dim = h_dim
        
    def forward(self, x,h_0,c_0, **kw):
        # x shape: (S, B) Note batch dim is not first!
        S, B = x.shape
        embedded = self.embedding(x) # embedded shape: (S, B, E)        
        out, (h_t, c_t) = self.lstm(embedded, (h_0, c_0))
        return (h_t, c_t)
class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, h_dim, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, h_dim, num_layers=num_layers, dropout=dropout)
        self.out_fc = nn.Linear(h_dim, vocab_size)
        
    def forward(self, x, context, **kw):
        S, B = x.shape
        embedded = self.embedding(x)
        output, (h_t, c_t) = self.lstm(embedded, context)
        out = self.out_fc(output)
        return out, (h_t, c_t)

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Seq2SeqEncoder, decoder: Seq2SeqDecoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder    
    def forward(self, x_src, x_tgt, p_tf=0, **kw):
        S2, B = x_tgt.shape        
        h_0 = torch.zeros(NUM_LAYERS, B, HID_DIM ) #(num_layers, batch_size, h_dim)
        c_0 = torch.zeros(NUM_LAYERS, B, HID_DIM ) #(num_layers, batch_size, h_dim)
        context = self.enc(x_src,h_0,c_0, **kw)        
        dec_input = x_tgt[[0], :]
        dec_outputs = []
        for t in range(1, S2):
            dec_output, context = self.dec(dec_input, context, **kw)           
            dec_outputs.append(dec_output)            
            if p_tf > torch.rand(1).item():
                dec_input = x_tgt[[t], :] 
            else:
                dec_input = torch.argmax(dec_output, dim=2)            
        y_hat = torch.cat(dec_outputs, dim=0)
        return y_hat
def init_mod(vocabSize,tgt):
    enc = Seq2SeqEncoder(vocabSize, EMB_DIM, NUM_LAYERS, HID_DIM)
    dec = Seq2SeqDecoder(tgt, EMB_DIM, NUM_LAYERS, HID_DIM)
    return Seq2Seq(enc, dec)