"""Non-personalized baselines for the MovieLens results table.

These exist so the GNN's results table answers a meaningful question: not
just "does the GNN predict ratings accurately" but "is the GNN actually
personalizing, or just learning popularity/averages". A GNN that doesn't
beat PopularityBaseline on ranking metrics would be a real, reportable
finding -- not a sign something is broken.
"""

import torch


class GlobalMeanBaseline:
    def fit(self, train_edge_label):
        self.mean = float(train_edge_label.mean())

    def predict(self, edge_label_index):
        return torch.full((edge_label_index.size(1),), self.mean)


class UserMeanBaseline:
    def fit(self, train_edge_label_index, train_edge_label, num_users):
        sums = torch.zeros(num_users)
        counts = torch.zeros(num_users)
        for u, r in zip(train_edge_label_index[0].tolist(), train_edge_label.tolist()):
            sums[u] += r
            counts[u] += 1
        self.global_mean = float(train_edge_label.mean())
        self.user_mean = torch.where(counts > 0, sums / counts.clamp(min=1), torch.full_like(sums, self.global_mean))

    def predict(self, edge_label_index):
        return self.user_mean[edge_label_index[0]]


class ItemMeanBaseline:
    def fit(self, train_edge_label_index, train_edge_label, num_movies):
        sums = torch.zeros(num_movies)
        counts = torch.zeros(num_movies)
        for m, r in zip(train_edge_label_index[1].tolist(), train_edge_label.tolist()):
            sums[m] += r
            counts[m] += 1
        self.global_mean = float(train_edge_label.mean())
        self.item_mean = torch.where(counts > 0, sums / counts.clamp(min=1), torch.full_like(sums, self.global_mean))

    def predict(self, edge_label_index):
        return self.item_mean[edge_label_index[1]]


class PopularityBaseline:
    """Ranks every user's candidates by global rating-count popularity,
    regardless of the user -- a non-personalized ranking baseline."""

    def fit(self, train_edge_label_index, num_movies):
        counts = torch.zeros(num_movies)
        for m in train_edge_label_index[1].tolist():
            counts[m] += 1
        self.popularity_score = counts

    def score_all_movies(self):
        return self.popularity_score
