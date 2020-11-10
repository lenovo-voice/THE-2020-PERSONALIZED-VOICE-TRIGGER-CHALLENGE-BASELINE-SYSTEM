import torch
import torch.nn as nn
from torch.autograd import Variable
# from modules.reverse_gradient import ReverseLayerF
import torch.nn.functional as F
import numpy as np
import random

def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze(dim=2)


def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze(dim=2)


def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        #print("h_i:",h_i.size())
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        #print("a_i:",a_i.size())
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 1)


class LSTMAtten(nn.Module):

    def __init__(self, input_dim=80, hidden_dim=128, output_dim=2, num_layers=2):
        super(LSTMAtten, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=0.2)
        self.weight_proj = nn.Parameter(torch.Tensor(hidden_dim, 1))
        self.weight_W = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim, 1))
        self.softmax = nn.Softmax(dim=1)
        self.weight_proj.data.uniform_(-0.1, 0.1)
        self.weight_W.data.uniform_(-0.1, 0.1)
        self.hidden2out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x
        out = out.reshape(out.shape[0], out.shape[2], out.shape[3])
        out = out.transpose(1,2)
        out, _ = self.lstm(out)
        #print("out.size() ",out.size())
        squish = batch_matmul_bias(out, self.weight_W, self.bias, nonlinearity='tanh')
        #print("squish.size() ",squish.size())
        attn = batch_matmul(squish, self.weight_proj)
        #print("attn.size() ",attn.size())
        attn_norm = self.softmax(attn.transpose(1,0))
        #print("attn_norm.size() ",attn_norm.size())
        attn_vectors = attention_mul(out, attn_norm.transpose(1,0))
        out = self.hidden2out(attn_vectors)
        return out, out

class LSTMAvg(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=128, output_dim=2, num_layers=2):
        super(LSTMAvg, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=0.2, batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #print(x.size())
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = x.transpose(1,2)
        #print(x.size())
        self.lstm.flatten_parameters()
        x, (ht, ct) = self.lstm(x)
        out = x.mean(1) # pooling
        out = self.hidden2out(out)
        return out, out

class LSTMLast(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=128, output_dim=2, num_layers=2):
        super(LSTMLast, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=0.2, batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = x.transpose(1,2)
        self.lstm.flatten_parameters()
        x, (ht, ct) = self.lstm(x)
        out = ht[-1] # last hidden
        out = self.hidden2out(out)
        return out, out

class LSTMLastDE(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=2, num_layers=2):
        super(LSTMLastDE, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=0.2, batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x1, x2):
        x1 = x1.reshape(x1.shape[0], x1.shape[2], x1.shape[3])
        x2 = x2.reshape(x2.shape[0], x2.shape[2], x2.shape[3])
        self.lstm.flatten_parameters()
        x1, (ht1, ct1) = self.lstm(x1)
        x2, (ht2, ct2) = self.lstm(x2)
        emb1 = ht1[-1] # last hidden
        emb2 = ht2[-1]
        out = self.hidden2out(emb1)
        return out, emb1, emb2


class LSTMLastGR(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=2, num_layers=2, output_spk_num=10, is_eval=False):
        super(LSTMLastGR, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=0.2, batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, output_dim)
        self.hidden2spk = nn.Linear(hidden_dim, output_spk_num)
        self.is_eval = is_eval

    def forward(self, x, alpha=None):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        self.lstm.flatten_parameters()
        x, (ht, ct) = self.lstm(x)
        out = ht[-1] # last hidden
        out_wrd = self.hidden2out(out)
        if self.is_eval or alpha == None:
            return out_wrd, out_wrd
        reverse_out = ReverseLayerF.apply(out, alpha)
        out_spk = self.hidden2spk(reverse_out)
        return out_wrd, out_spk

class LSTMTrans(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=2, num_layers=2):
        super(LSTMTrans, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=0.2, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.hidden2out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, alpha=None):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        self.lstm.flatten_parameters()
        x, (ht, ct) = self.lstm(x)
        x = self.transformer_encoder(x)
        x = x.mean(1)
        x = self.hidden2out(x)
        return x, x

class LSTMTrans1_deep(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=2, num_layers=2):
        super(LSTMTrans1_deep, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=0.2, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.hidden2out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, alpha=None):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        self.lstm.flatten_parameters()
        x, (ht, ct) = self.lstm(x)
        x = self.transformer_encoder(x)
        x = x.mean(1)
        x = self.hidden2out(x)
        return x, x

class LSTMTrans2_swc(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=2, num_layers=2):
        super(LSTMTrans2_swc, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=1, dropout=0.2, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, dropout=0.2, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1)
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.hidden2out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, alpha=None):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        x, (ht, ct) = self.lstm1(x)
        x = self.transformer_encoder1(x)
        # print(x.shape)
        x, (ht, ct) = self.lstm2(x)
        x = self.transformer_encoder2(x)
        x = x.mean(1)
        x = self.hidden2out(x)
        return x, x

def MainModel(nOut=4):
    model = LSTMAvg(nOut)
    return model

def test():
    net = LSTMAtten()
    a = torch.rand(128, 1, 80, 121)
    b, _ = net(a)
    print(b.shape)
    count = 0
    for p in net.parameters():
        count += p.data.nelement()
    print("param:", count)

if __name__ == "__main__":
    test()
