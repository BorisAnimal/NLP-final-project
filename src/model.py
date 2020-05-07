import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, word_vocab_size, word_embedding_dim, hidden_dim, target_size, padding_idx,
                 fc1=128, dropout_rate=0.1, topk=3):
        super(Classifier, self).__init__()
        self.topk = topk
        self.emb = nn.Embedding(word_vocab_size, word_embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(word_embedding_dim, hidden_dim, bidirectional=True)  # , num_layers=1, dropout=dropout_rate)
        self.clf = nn.Sequential(
            nn.Linear(hidden_dim * 4 * self.topk, fc1),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(fc1, target_size)
        )

        self.init_weights()

    def forward(self, tokens):
        h = self.emb(tokens)
        h, lstm_state = self.lstm(h)
        maxp = torch.topk(h, self.topk, 1, largest=True)[0].view((h.shape[0], -1))
        minp = torch.topk(h, self.topk, 1, largest=False)[0].view((h.shape[0], -1))
        h = torch.cat([maxp, minp], 1)
        return self.clf(h)

    def init_weights(self):
        modules = self.modules()
        for i, m in enumerate(modules):
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0.01)
            if isinstance(m, nn.Linear):
                # torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

