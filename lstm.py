import torch
import torch.nn as nn

## Input shape: (batch_size * num_mention * num_candidate * max_length_word, max_length_char)
## Output shape: (batch_size * num_mention * num_candidate * max_length_word, max_length_char * hidden_size)

class EmbedCharLayer(nn.Module):
    def __init__(self, vocab_size, max_length_seq_char, input_dim, hidden_size, num_layer, device='cpu', dropout=0.2, bidirection=True):
        super().__init__()
        self.input_dim = input_dim 
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.device = device
        self.max_length_seq_char = max_length_seq_char
        self.bidirection = bidirection
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layer, batch_first=True, dropout=dropout, bidirectional=bidirection)

    def forward(self, x):
        total_size = 1
        for s in x.shape:
            total_size = total_size * s

        x = self.embedding(x)  
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        ho, co = self.init_lstm_state(total_size // self.max_length_seq_char, self.num_layer, self.hidden_size, self.device, bidirection=self.bidirection)
        x, (hn, cn) = self.lstm(x, (ho, co))

        return x


    def init_lstm_state(self, batch_size, num_layer, hidden_size, device='cpu', bidirection=True):
        if bidirection:
            return (torch.rand((2 * num_layer, batch_size, hidden_size), device=device), torch.rand((2 * num_layer,batch_size, hidden_size), device=device))
        else:
            return (torch.rand((num_layer, batch_size, hidden_size), device=device), torch.rand((num_layer, batch_size, hidden_size), device=device))

## Input shape: (batch_size, max_length_sence)
## Output shape: (batch_size, hidden_size)
class EmbedSentenceLayer(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_size, num_layer, device='cpu', dropout=0.2, bidirection=True):
        super().__init__()
        self.input_dim = input_dim 
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.device = device
        self.bidirection = bidirection
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layer, batch_first=True, dropout=dropout, bidirectional=bidirection)

    def forward(self, x):
        x = self.embedding(x)
        batch_size = x.shape[0]
        # print(x.shape)
        ho, co = self.init_lstm_state(batch_size, self.num_layer, self.hidden_size, device=self.device)
        x, (_, _) = self.lstm(x, (ho, co))
        x = self.averge_embed(x, batch_size)

        return x

    def init_lstm_state(self, batch_size, num_layer, hidden_size, device='cpu', bidirection=True):
        if bidirection:
            return (torch.rand((2 * num_layer, batch_size, hidden_size), device=device), torch.rand((2 * num_layer, batch_size, hidden_size), device=device))
        else:
            return (torch.rand((num_layer, batch_size, hidden_size), device=device), torch.rand((num_layer, batch_size, hidden_size), device=device))

    def averge_embed(self, x, batch_size):
    
        return torch.mean(x, dim=1, keepdim=True).reshape(batch_size, -1)