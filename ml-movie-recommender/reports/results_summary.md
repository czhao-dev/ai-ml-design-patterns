# Results Summary

## IMDb track (rating prediction, sample data)

| Method | Batman v Superman: Dawn of Justice | Mission: Impossible - Rogue Nation | Minions |
|---|---|---|---|
| Neighborhood Averaging | 7.05 | 7.70 | 6.30 |
| Linear Regression | 6.83 | 7.40 | 6.40 |
| Bipartite Graph Averaging | 6.97 | 7.60 | 6.40 |
| GNN (heterogeneous, LOO) | 6.74 | 7.72 | 5.96 |
| **IMDb (sample data)** | 6.40 | 7.40 | 6.40 |

GNN leave-one-out CV across all 7 labeled sample movies: RMSE=1.1561, MAE=0.9118. The three heuristic baselines only ever produced predictions for the three demo movies shown above (that's all the original pipeline computed), so no comparable full-sample RMSE/MAE exists for them -- only the GNN's LOO-CV covers all 7 labeled movies.

**Methodology note**: every number above is an out-of-sample prediction (leave-one-out cross-validation -- each movie's prediction comes from a model that never saw that movie's rating during training). On a sample this small (7 labeled movies), a strong-looking RMSE only shows the pipeline runs correctly end-to-end, not that the architecture works at scale -- a GNN can memorize 7 points as easily as the existing LinearRegression baseline overfits them on this same sample (R^2 ~= 0.42 on sample vs ~= 0.005 on the full, unincluded IMDb dataset).
