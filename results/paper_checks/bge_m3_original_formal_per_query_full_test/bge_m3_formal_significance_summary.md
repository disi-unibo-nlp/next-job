# BGE-M3 Formal Significance Test

Paired bootstrap with 10000 resamples, seed=42. Delta = formal BGE-M3+Markov fusion - formal BGE-M3 pure.

| Metric | N | Pure | Fusion | Delta | 95% CI | p_boot |
|---|---:|---:|---:|---:|---|---:|
| Accuracy@1 | 32089 | 0.034373 | 0.077659 | 0.043286 | [0.039827, 0.046714] | 0 |
| MRR@10 | 32089 | 0.068549 | 0.139675 | 0.071126 | [0.067961, 0.074330] | 0 |
| NDCG@10 | 32089 | 0.089046 | 0.179024 | 0.089978 | [0.086715, 0.093220] | 0 |
| Recall@10 | 32089 | 0.155412 | 0.307146 | 0.151734 | [0.147060, 0.156409] | 0 |
