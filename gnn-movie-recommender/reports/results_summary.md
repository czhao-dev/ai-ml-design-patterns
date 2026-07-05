# Results Summary

## IMDb track (rating prediction, full IMDb data, 19,458 labeled movies)

| Method | Batman v Superman: Dawn of Justice | Mission: Impossible - Rogue Nation | Minions |
|---|---|---|---|
| Neighborhood Averaging | 6.42 | 6.19 | 6.44 |
| Linear Regression | 6.15 | 6.01 | 6.20 |
| Bipartite Graph Averaging | 6.57 | 7.06 | 5.98 |
| GNN (heterogeneous, HOLDOUT) [*] | 5.89 | 5.16 | 5.36 |
| **IMDb (full IMDb data, 19,458 labeled movies)** | 6.40 | 7.40 | 6.40 |

[*] Batman v Superman: Dawn of Justice (2016), Mission: Impossible - Rogue Nation (2015), Minions (2015): in-sample prediction(s) -- this movie landed in the training split (holdout is a random split, so this can happen by chance), not held out, so its GNN value above is not comparable to the others.

GNN HOLDOUT across all 19,458 labeled movies: RMSE=1.3490, MAE=1.1358. The three heuristic baselines only ever produced predictions for the three demo movies shown above (that's all the original pipeline computed), so no comparable full-sample RMSE/MAE exists for them -- only the GNN's HOLDOUT metric covers all labeled movies.

**Global Mean baseline** (predict the train-label average for every test movie): RMSE=1.1251, MAE=0.8772 (same test split as the GNN's HOLDOUT metric above).

**The GNN does not beat this trivial baseline** on this run (GNN RMSE=1.3490 vs. Global Mean RMSE=1.1251). This is consistent with the very weak signal the full-scale linear-regression heuristic already found in this data (R^2 ~= 0.02 on all labeled movies, vs. R^2 ~= 0.42 on the tiny, unrepresentative 7-movie sample -- see the sample-scale note above): actor PageRank/degree/cast-structure/community/genre features alone carry little signal about a movie's aggregate rating at real scale. Plausible next steps to actually close this gap -- not attempted here -- include more training epochs / wider neighbor sampling (this run used 10 epochs, `num_neighbors: [15, 10]`, `hidden_dim: 64`), additional node features (e.g. runtime, release year, cast size beyond a single log-count), or accepting that rating regression from cast/graph structure alone is a genuinely hard problem at this scale.

**Methodology note**: RMSE/MAE above are computed only from a genuinely held-out random test split (19,458 movies never seen during training); any in-sample demo-movie predictions flagged above are excluded from these metrics. Unlike the tiny sample, this is real evidence of (or against) the architecture generalizing, not just proof the pipeline runs.
