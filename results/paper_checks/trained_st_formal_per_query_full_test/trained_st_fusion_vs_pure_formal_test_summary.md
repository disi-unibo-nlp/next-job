# trained ST+Markov fusion vs trained ST pure

Paired bootstrap with 10000 resamples, seed=42. Delta = system - baseline.

| Metric | N | Baseline | System | Delta | 95% CI | p_boot |
|---|---:|---:|---:|---:|---|---:|
| Accuracy@1 | 32089 | 0.052604 | 0.086385 | 0.033781 | [0.030197, 0.037396] | 0 |
| MRR@10 | 32089 | 0.110855 | 0.154709 | 0.043854 | [0.040950, 0.046731] | 0 |
| NDCG@10 | 32089 | 0.149258 | 0.198201 | 0.048943 | [0.046170, 0.051652] | 0 |
| Recall@10 | 32089 | 0.275016 | 0.339836 | 0.064820 | [0.060862, 0.068747] | 0 |
