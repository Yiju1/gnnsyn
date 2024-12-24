# gnnsyn/model/temporal.py

import torch.nn as nn

class RNNTemporal(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type='rnn'):
        super().__init__()
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

    def forward(self, x_seq):
        """
        x_seq: [batch_size, seq_len, input_size]
        """
        out, _ = self.rnn(x_seq)
        # 只取最后一个时间步
        return out[:, -1, :]
