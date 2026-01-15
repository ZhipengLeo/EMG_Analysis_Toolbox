import torch
import torch.nn as nn
import torch.nn.functional as F

class EMGCNN(nn.Module):
    """
    基于特征图的 CNN 手势识别模型

    输入:
        x: (batch_size, C, F)
    输出:
        logits: (batch_size, n_classes)
    """

    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=n_channels,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            shape = (B, C, F)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.pool(x)          # (B, 128, 1)
        x = x.squeeze(-1)         # (B, 128)

        x = self.fc(x)            # (B, n_classes)
        return x
