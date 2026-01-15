import torch
import torch.nn as nn

class EMGLSTM(nn.Module):
    """
    基于特征序列的 LSTM 手势识别模型

    输入:
        x: (batch_size, F, C)
    输出:
        logits: (batch_size, n_classes)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        num_layers=1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            shape = (B, F, C)
        """
        _, (h_n, _) = self.lstm(x)

        h_last = h_n[-1]           # (B, hidden_size)
        out = self.fc(h_last)     # (B, num_classes)

        return out
