from gnn.metrics.ranking import ndcg_at_k, precision_at_k, recall_at_k
from gnn.metrics.regression import mae, r_squared, rmse


def test_rmse_mae_perfect_predictions():
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_rmse_mae_known_values():
    preds = [1.0, 2.0, 3.0]
    targets = [2.0, 2.0, 5.0]
    # errors: -1, 0, -2 -> squared: 1, 0, 4 -> mean 5/3
    assert abs(rmse(preds, targets) - (5 / 3) ** 0.5) < 1e-9
    assert abs(mae(preds, targets) - 1.0) < 1e-9


def test_r_squared_perfect_fit():
    assert r_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_precision_recall_at_k():
    # top-3 of 5 ranked items: relevant, not, relevant, not, not
    flags = [1, 0, 1, 0, 0]
    assert precision_at_k(flags, 3) == 2 / 3
    assert recall_at_k(flags, 3, num_relevant=2) == 1.0
    assert recall_at_k(flags, 1, num_relevant=2) == 0.5


def test_recall_at_k_no_relevant_items():
    assert recall_at_k([0, 0, 0], 3, num_relevant=0) == 0.0


def test_ndcg_at_k_perfect_ranking():
    # best possible ranking: all relevant items first -> NDCG == 1
    assert abs(ndcg_at_k([1, 1, 0, 0], 4) - 1.0) < 1e-9


def test_ndcg_at_k_no_relevant_items():
    assert ndcg_at_k([0, 0, 0], 3) == 0.0
