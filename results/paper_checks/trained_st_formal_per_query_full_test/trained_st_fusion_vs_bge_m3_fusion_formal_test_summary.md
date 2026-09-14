# trained ST+Markov fusion vs BGE-M3+Markov fusion

Paired bootstrap with 10000 resamples, seed=42. Delta = system - baseline.

| Metric | N | Baseline | System | Delta | 95% CI | p_boot |
|---|---:|---:|---:|---:|---|---:|
| Accuracy@1 | 32089 | 0.077659 | 0.086385 | 0.008726 | [0.006918, 0.010564] | 0 |
| MRR@10 | 32089 | 0.139675 | 0.154709 | 0.015034 | [0.013619, 0.016457] | 0 |
| NDCG@10 | 32089 | 0.179024 | 0.198201 | 0.019177 | [0.017541, 0.020786] | 0 |
| Recall@10 | 32089 | 0.307146 | 0.339836 | 0.032690 | [0.029168, 0.036149] | 0 |
