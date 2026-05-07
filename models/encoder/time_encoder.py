import torch
import torch.nn as nn
import torch.nn.functional as F


class time_encoder(nn.Module):
    def __init__(self, input_channels, num_hidden, embedding_dimension, kernel_size=3, stride=1, dropout=0):
        super(time_encoder, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size // 2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden * 2, kernel_size=kernel_size, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(num_hidden * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(num_hidden * 2, embedding_dimension, kernel_size=kernel_size, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(embedding_dimension),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

    def forward(self, x):
        # print('input size is {}'.format(x_in.size()))

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # print(x.size())
        return x


# Add GRU==========================================================================================
class GRU_Time_Feature_Extractor(nn.Module):
    def __init__(self, input_channels, num_hidden, embedding_dimension, kernel_size, stride=1, dropout=0.3):
        super(GRU_Time_Feature_Extractor, self).__init__()

        # Set padding
        paddings = [(k - 1) // 2 for k in kernel_size]

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size[0],
                      stride=stride, bias=False, padding=paddings[0]),
            nn.BatchNorm1d(num_hidden),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden * 2, kernel_size=kernel_size[1], stride=1,
                      bias=False, padding=paddings[1]),
            nn.BatchNorm1d(num_hidden * 2),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(num_hidden * 2, num_hidden * 2, kernel_size=kernel_size[2], stride=1,
                      bias=False, padding=paddings[2]),
            nn.BatchNorm1d(num_hidden * 2),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.gru = nn.GRU(input_size=num_hidden * 2, hidden_size=embedding_dimension, batch_first=True)


    def forward(self, x):
        """
        :param x: [batch_size, sequence_length, dims]
        :return: [batch_size, embedding_dimension]
        """

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = torch.transpose(x, 1, 2)  # [batch_size, sequence_length, embedding_dimension]

        self.gru.flatten_parameters()
        x, _ = self.gru(x)  # x shape: [batch_size, sequence_length, embedding_dimension]

        return x