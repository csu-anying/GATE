import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange

from models.encoder.gcn import GCN, GIN
from models.augmentors.augs import  augmentor_only_time_feature, create_augmentor_node_only, create_augmentor_edge_only, create_augmentor_both
from models.encoder.time_encoder import GRU_Time_Feature_Extractor


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(100.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        # return x


class Dot_Graph_Construction_weights(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, node_features):
        device = node_features.device
        node_features = self.mapping(node_features)
        # node_features = F.leaky_relu(node_features)
        bs, N, dimen = node_features.size()

        node_features_1 = torch.transpose(node_features, 1, 2)

        Adj = torch.bmm(node_features, node_features_1)

        eyes_like = torch.eye(N).repeat(bs, 1, 1).to(device)
        eyes_like_inf = eyes_like * 1e8
        Adj = F.leaky_relu(Adj - eyes_like_inf)
        Adj = F.softmax(Adj, dim=-1)
        # print("adj size", Adj.size())
        Adj = Adj + eyes_like
        # print(Adj[0])
        # if prior:

        return Adj


class Graph_Encoder(nn.Module):
    def __init__(self, network_config):
        super(Graph_Encoder, self).__init__()
        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']
        self.proj_out = network_config['projection_head']['projection_size']
        self.time_out = network_config['projection_head']['time_out_size']
        self.patch_size = network_config['patch_size']
        # layers
        self.time_feature_encoder = GRU_Time_Feature_Extractor(1, self.proj_hidden, self.time_out, kernel_size=[9, 7, 5])
        self.nonlin_map = nn.Sequential(
            nn.Linear(self.time_out * 64, self.time_out),
            nn.BatchNorm1d(self.time_out)
        )
        self.positional_encoding = PositionalEncoding(self.time_out, 0.1, max_len=5000)
        self.graph_construction = Dot_Graph_Construction_weights(self.time_out)

        self.graph_learning = GIN(self.time_out, self.proj_out, k=1)


        self.augmentor = create_augmentor_both(node_mask_ratio=0.2, edge_keep_ratio=0.8, edge_perturb_ratio=0.3,
                                                    seed=42)

        # Only node
        # self.augmentor = create_augmentor_node_only(node_mask_ratio=0.2, edge_keep_ratio=0.8, edge_perturb_ratio=0.3,
        #                                             seed=42)
        # Only edge
        # self.augmentor = create_augmentor_edge_only(node_mask_ratio=0.2, edge_keep_ratio=0.8, edge_perturb_ratio=0.3,
        #                                             seed=42)


    def patchify(self, series):
        """
        series: (batch_size, num_leads, seq_len)
        x: (batch_size, num_leads, n, patch_size)
        """
        p = self.patch_size
        assert series.shape[2] % p == 0
        x = rearrange(series, 'b c (n p) -> b c n p', p=p)
        return x

    def forward(self, X, mode):
        ecg = self.patchify(X)  # b c n p
        batch_size, num_leads, tlen, patch_size = ecg.size()
        # Graph Generation
        A_input = torch.reshape(ecg, [batch_size * tlen * num_leads, patch_size, 1])
        A_input = A_input.transpose(1, 2)
        A_input_ = self.time_feature_encoder(A_input)

        # positional encodeing
        X_ = torch.reshape(A_input_, [batch_size * tlen * num_leads, -1])
        X_ = self.nonlin_map(X_)
        # temp = torch.reshape(X_, [batch_size, -1])
        # return temp

        X_ = torch.reshape(X_, [batch_size * num_leads, tlen, -1])
        X_ = self.positional_encoding(X_)

        # build graph
        X_ = torch.reshape(X_, [batch_size, tlen * num_leads, -1])  # [32,600,128]
        adj = self.graph_construction(X_)  # [32,600,600]

        X = F.relu(self.graph_learning(X_, adj))
        X = torch.mean(X, dim=1)

        if mode == "train":
            # Augmentation
            X1, adj1, X2, adj2 = self.augmentor.forward(X_, adj)

            # Graph Learning on augmented views
            X_aug1 = self.graph_learning(X1, adj1)
            X_aug2 = self.graph_learning(X2, adj2)

            # Global pooling to get graph-level representations
            X_aug1 = torch.mean(X_aug1, dim=1)
            X_aug2 = torch.mean(X_aug2, dim=1)

            return X, X_aug1, X_aug2

        return X


# =======================================time feature only===============
class Graph_Encoder_time(nn.Module):
    def __init__(self, network_config):
        super(Graph_Encoder_time, self).__init__()
        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']
        self.proj_out = network_config['projection_head']['projection_size']
        self.time_out = network_config['projection_head']['time_out_size']
        self.patch_size = network_config['patch_size']

        # Time Feature Extraction
        self.time_feature_encoder = GRU_Time_Feature_Extractor(1, self.proj_hidden, self.time_out, kernel_size=[9, 7, 5])

        self.nonlin_map = nn.Sequential(
            nn.Linear(self.time_out * 64, self.proj_out),
            nn.BatchNorm1d(self.proj_out)
        )
        self.augmentor = augmentor_only_time_feature(node_mask_ratio=0.2, seed=42)

    def patchify(self, series):
        p = self.patch_size
        assert series.shape[2] % p == 0
        x = rearrange(series, 'b c (n p) -> b c n p', p=p)
        return x

    def forward(self, X, mode):
        ecg = self.patchify(X)  # b c n p
        batch_size, num_leads, tlen, patch_size = ecg.size()
        A_input = torch.reshape(ecg, [batch_size * tlen * num_leads, patch_size, 1])
        A_input = A_input.transpose(1, 2)
        A_input_ = self.time_feature_encoder(A_input)
        X_ = torch.reshape(A_input_, [batch_size * tlen * num_leads, -1])
        # print("X3", X_.shape)
        X_ = self.nonlin_map(X_)
        X_ = torch.reshape(X_, [batch_size, tlen * num_leads, -1])

        X = torch.mean(X_, dim=1)

        if mode == "train":
            X_aug1, X_aug2 = self.augmentor.forward(X)

            return X, X_aug1, X_aug2

        return X