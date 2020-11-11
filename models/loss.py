# Author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import LpDistance


def make_loss(params):
    if params.loss == 'BatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = BatchHardTripletLossWithMasks(params.margin, params.normalize_embeddings)
    elif params.loss == 'BatchHardContrastiveLoss':
        loss_fn = BatchHardContrastiveLossWithMasks(params.pos_margin, params.neg_margin, params.normalize_embeddings)
    else:
        print('Unknown loss: {}'.format(params.loss))
        raise NotImplementedError
    return loss_fn


class HardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance):
        self.distance = distance
        # Stats
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, embeddings, positives_mask, negatives_mask):
        assert embeddings.dim() == 2
        d_embeddings = embeddings.detach()
        with torch.no_grad():
            hard_triplets = self.mine(d_embeddings, positives_mask, negatives_mask)
        return hard_triplets

    def mine(self, embeddings, positives_mask, negatives_mask):
        # Based on pytorch-metric-learning implementation
        dist_mat = self.distance(embeddings)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        self.max_pos_pair_dist = torch.max(hardest_positive_dist).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist).item()
        return a, p, n


def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows


class BatchHardTripletLossWithMasks:
    def __init__(self, margin, normalize_embeddings):
        self.margin = margin
        self.normalize_embeddings = normalize_embeddings
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings)
        # We use triplet loss with Euclidean distance
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=True, distance=self.distance)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
                 'num_triplets': len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats, hard_triplets


class BatchHardContrastiveLossWithMasks:
    def __init__(self, pos_margin, neg_margin, normalize_embeddings):
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        # We use contrastive loss with squared Euclidean distance
        self.loss_fn = losses.ContrastiveLoss(pos_margin=self.pos_margin, neg_margin=self.neg_margin,
                                              distance=self.distance)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'pos_pairs_above_threshold': self.loss_fn.reducer.reducers['pos_loss'].pos_pairs_above_threshold,
                 'neg_pairs_above_threshold': self.loss_fn.reducer.reducers['neg_loss'].neg_pairs_above_threshold,
                 'pos_loss': self.loss_fn.reducer.reducers['pos_loss'].pos_loss.item(),
                 'neg_loss': self.loss_fn.reducer.reducers['neg_loss'].neg_loss.item(),
                 'num_pairs': 2*len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats, hard_triplets
