# Models

Trained model artifacts (baseline response model, T-learner base learners, X-learner base/effect
learners, causal forest) are saved locally under:

```text
models/trained/
```

That folder is ignored by Git — every model is fast to retrain from `data/raw/hillstrom.csv` via
the numbered scripts in `scripts/`, so nothing here needs to be committed or shared externally.
