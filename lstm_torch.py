import torch.nn as nn
from torch.autograd import Variable

class LM_LSTM(nn.Module):
  """Simple LSMT-based language model"""
  def __init__(self, embedding_dim,  vocab_size, num_layers,hidden_dim):
    super(LM_LSTM, self).__init__()
    self.embedding_dim = embedding_dim
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.hidden_dim = hidden_dim
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers)
    self.sm_fc = nn.Linear(hidden_dim,vocab_size)
    self.hidden_layer=init_hidden()

 
  def init_hidden(self):
    return (autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)))
  def forward(self, inputs, hidden):
    embeds = self.dropout(self.word_embeddings(inputs))
    lstm_out, hidden = self.lstm(embeds, hidden)
    lstm_out = self.dropout(lstm_out)
    logits = self.sm_fc(lstm_out.view(-1, self.embedding_dim))
    return logits.view(self.num_steps, self.batch_size, self.vocab_size), hidden
