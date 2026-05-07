import os
from typing import Type
import torch
import torch.nn.functional as F
import pandas as pd
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm


def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
    ''' Compute the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def clip_loss(x, y, temperature=0.2, device='cuda'):
    """
    Compute CLIP loss for inter-modal contrastive learning
    x: [batch_size, feature_dim] - Features of the original view
    y: [batch_size, feature_dim] - Enhanced view features
    """
    # Avoid in-place operations
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)

    sim = torch.einsum('i d, j d -> i j', x_norm, y_norm) * 1 / temperature

    labels = torch.arange(x.shape[0]).to(device)

    loss_t = F.cross_entropy(sim, labels)
    loss_i = F.cross_entropy(sim.T, labels)

    i2t_acc1, i2t_acc5 = precision_at_k(
        sim, labels, top_k=(1, 5))
    t2i_acc1, t2i_acc5 = precision_at_k(
        sim.T, labels, top_k=(1, 5))
    acc1 = (i2t_acc1 + t2i_acc1) / 2.
    acc5 = (i2t_acc5 + t2i_acc5) / 2.

    return (loss_t + loss_i), acc1, acc5


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


_dot = lambda x, y: x @ y.t()


def compute_infonce(anchor, sample, pos_mask, tau=0.2, *args, **kwargs):
    """
    @pos_mask: indicate postive samples, 默认为h1[i], h2[i]
        default: eye(auchor.shape[0])
        extra_pos: eye(auchor.shape[0]) + supervised pos, h1[i] and h2[j] j in same class.
    @example: 0.5*compute_infonce(h1, h2, pos_mask) + 0.5*compute_infonce(h2, h1, pos_mask)
    """
    sim = _similarity(anchor, sample) / tau  # [len(h1), len(h2)]
    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # [len(h1), len(h2)] - [len(h1), 1]
    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
    return -loss.mean()


def graph_contrastive_loss(x1, x2, temperature=0.2):
    """
    x1, x2: Graph embedding of two enhanced views, shape: [batch_size, hidden_dim]
    temperature: Temperature parameter, used for scaling similarity
    """
    batch_size = x1.size(0)

    # Normalization (necessary: cosine similarity)
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)

    # Similarity matrix: [B, B]
    sim_matrix = torch.mm(x1, x2.t()) / temperature  # Cosine similarity + temperature scaling

    # log-softmax loss
    loss_i = F.cross_entropy(sim_matrix, torch.arange(batch_size).to(x1.device))
    loss_j = F.cross_entropy(sim_matrix.t(), torch.arange(batch_size).to(x1.device))

    loss = (loss_i + loss_j) / 2
    return loss


def cosine_consistency_loss(proj_ecg_emb, ecg_graph_emb1, ecg_graph_emb2):
    loss1 = 1 - F.cosine_similarity(proj_ecg_emb, ecg_graph_emb1, dim=-1).mean()
    loss2 = 1 - F.cosine_similarity(proj_ecg_emb, ecg_graph_emb2, dim=-1).mean()
    return (loss1 + loss2) / 2
