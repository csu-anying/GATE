import torch
import random
import numpy as np

class GraphAugmentor:
    def __init__(self, node_mask_ratio=0.2, edge_keep_ratio=0.7, edge_perturb_ratio=0.3,
                 use_node_mask=True, use_edge_perturb=True, seed=42):
        self.node_mask_ratio = node_mask_ratio
        self.edge_keep_ratio = edge_keep_ratio
        self.edge_perturb_ratio = edge_perturb_ratio
        self.use_node_mask = use_node_mask
        self.use_edge_perturb = use_edge_perturb
        self.seed = seed
        self.set_seed(seed)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def mask_node_features(self, X):
        B, N, F = X.shape
        num_mask = int(N * self.node_mask_ratio)
        rand = torch.rand(B, N, device=X.device)
        mask = torch.ones_like(rand)
        _, indices = torch.topk(rand, num_mask, dim=1, largest=False)
        mask.scatter_(1, indices, 0)
        X_masked = X * mask.unsqueeze(-1)
        return X_masked, mask

    def feature_masking_elementwise(self, X, mask_prob=0.2):
        B, N, F = X.shape
        mask = torch.bernoulli((1 - mask_prob) * torch.ones_like(X)).to(X.device)
        return X * mask

    def edge_soft_perturb_vectorized(self, adj):
        B, N, _ = adj.shape
        triu_idx = torch.triu_indices(N, N, offset=1, device=adj.device)
        row_idx, col_idx = triu_idx[0], triu_idx[1]
        num_edges = row_idx.size(0)
        weights = adj[:, row_idx, col_idx]
        num_keep = int(num_edges * self.edge_keep_ratio)
        sorted_weights, sorted_indices = torch.topk(weights, num_keep, dim=1)
        full_mask = torch.zeros_like(weights, dtype=torch.bool)
        full_mask.scatter_(1, sorted_indices, True)
        drop_mask = ~full_mask
        noise = torch.rand_like(weights) * drop_mask * self.edge_perturb_ratio
        new_weights = weights * full_mask + weights * drop_mask * noise
        adj[:, row_idx, col_idx] = new_weights
        adj[:, col_idx, row_idx] = new_weights
        return adj

    def forward(self, X, adj):
        self.set_seed(self.seed + 1)
        X1 = self.feature_masking_elementwise(X) if self.use_node_mask else X
        A1 = self.edge_soft_perturb_vectorized(adj.clone()) if self.use_edge_perturb else adj

        self.set_seed(self.seed + 2)
        X2 = self.feature_masking_elementwise(X) if self.use_node_mask else X
        A2 = self.edge_soft_perturb_vectorized(adj.clone()) if self.use_edge_perturb else adj

        return X1, A1, X2, A2


# ====================================Only Time Features=======================================
class TimeAugmentor:
    def __init__(self, feature_mask_ratio=0.2, seed=42):
        self.feature_mask_ratio = feature_mask_ratio
        self.seed = seed
        self.dropout = torch.nn.Dropout(p=self.feature_mask_ratio)
        self.set_seed(seed)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def mask_features(self, X):
        """
        X: [B, F] — use dropout to randomly drop features
        """
        return self.dropout(X)  # no explicit mask returned

    def forward(self, X):
        self.set_seed(self.seed + 1)
        X1 = self.mask_features(X)

        self.set_seed(self.seed + 2)
        X2 = self.mask_features(X)

        return X1, X2


def create_augmentor_both(node_mask_ratio=0.2, edge_keep_ratio=0.7, edge_perturb_ratio=0.3, seed=42, use_node_mask=True,
                          use_edge_perturb=True):
    return GraphAugmentor(node_mask_ratio, edge_keep_ratio, edge_perturb_ratio, use_node_mask,
                                   use_edge_perturb, seed)


def create_augmentor_node_only(node_mask_ratio=0.2, edge_keep_ratio=0.7, edge_perturb_ratio=0.3, seed=42,
                               use_node_mask=True, use_edge_perturb=False):
    return GraphAugmentor(node_mask_ratio, edge_keep_ratio, edge_perturb_ratio, use_node_mask,
                                   use_edge_perturb, seed)


def create_augmentor_edge_only(node_mask_ratio=0.2, edge_keep_ratio=0.7, edge_perturb_ratio=0.3, seed=42,
                               use_node_mask=False, use_edge_perturb=True):
    return GraphAugmentor(node_mask_ratio, edge_keep_ratio, edge_perturb_ratio, use_node_mask,
                                   use_edge_perturb, seed)


def augmentor_only_time_feature(node_mask_ratio=0.2, seed=42):
    return TimeAugmentor(node_mask_ratio, seed)
