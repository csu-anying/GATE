import os
import sys

from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.encoder.graph_encoder import Graph_Encoder, Graph_Encoder_time


class ECG_Graph_Classifier(nn.Module):
    def __init__(self, network_config, num_classes):
        super(ECG_Graph_Classifier, self).__init__()

        self.graph_encoder = Graph_Encoder(network_config)
        # self.graph_encoder = Graph_Encoder_time(network_config)
        graph_out_dim = network_config['projection_head']['projection_size']
        self.classifier = nn.Sequential(
            nn.Linear(graph_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def extract_classifier_features(self, ecg):
        x = self.graph_encoder(ecg, mode='test')
        # x = ecg
        # x = self.classifier[0](x)  # Linear
        # x = self.classifier[1](x)  # ReLU
        b, _, _ = ecg.size()
        x = x.reshape(b, -1)
        return x  # shape: [B, 128]

    def forward(self, ecg):
        x = self.graph_encoder(ecg, mode='test')  # [B, N, D]
        logits = self.classifier(x)  # [B, num_classes]
        return logits
