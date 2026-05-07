import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class GIN(nn.Module):
    def __init__(self, input_dimension, output_dimension, k):
        # k in GIN is typically the number of layers
        super(GIN, self).__init__()
        self.k = k
        self.eps = nn.Parameter(torch.zeros(k))
        self.mlp = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dimension, output_dimension),
            nn.ReLU(),
            nn.Linear(output_dimension, output_dimension)
        ) for _ in range(k)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(output_dimension) for _ in range(k)])

    def norm(self, adj, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[..., idx, idx] += 1

        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        return adj

    def forward(self, X, A):
        # size of X is (bs, N, D)
        # size of A is (bs, N, N)
        A = self.norm(A, add_loop=False)
        h = X
        for i in range(self.k):
            h = self.mlp[i]((1 + self.eps[i]) * h + torch.bmm(A, h))
            h = torch.transpose(h, -1, -2)
            h = self.batch_norms[i](h)
            h = torch.transpose(h, -1, -2)
        return h
