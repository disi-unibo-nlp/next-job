# Significance Tests

Paired bootstrap with 10000 resamples, seed=42. Delta = system - baseline.

For Qwen ablations, per-query metrics are computed from `gold_rank_before_qwen` and `gold_rank_after_qwen`.

| Test | Metric | N | Baseline | System | Delta | 95% CI | p_boot | p_t_approx |
|---|---|---:|---:|---:|---:|---|---:|---:|
| wiki_kg_fusion_vs_kg_alone_test | Accuracy@1 | 32089 | 0.069744 | 0.072579 | 0.002836 | [0.001776, 0.003864] | 0 | 7.70546e-08 |
| wiki_kg_fusion_vs_kg_alone_test | MRR@10 | 32089 | 0.133072 | 0.137745 | 0.004673 | [0.003959, 0.005389] | 0 | 1.11501e-36 |
| wiki_kg_fusion_vs_kg_alone_test | NDCG@10 | 32089 | 0.176111 | 0.181313 | 0.005202 | [0.004320, 0.006081] | 0 | 6.89027e-31 |
| wiki_kg_fusion_vs_kg_alone_test | Recall@10 | 32089 | 0.317430 | 0.323943 | 0.006513 | [0.004238, 0.008757] | 0 | 1.75145e-08 |
| qwen_direct_select_vs_baseline_validation_n2000 | Accuracy@1 | 2000 | 0.071000 | 0.043500 | -0.027500 | [-0.039000, -0.016500] | 0 | 1.35068e-06 |
| qwen_direct_select_vs_baseline_validation_n2000 | MRR@10 | 2000 | 0.133909 | 0.102407 | -0.031502 | [-0.040507, -0.022693] | 0 | 5.57988e-12 |
| qwen_direct_select_vs_baseline_validation_n2000 | NDCG@10 | 2000 | 0.175972 | 0.149313 | -0.026659 | [-0.034064, -0.018979] | 0 | 7.83817e-12 |
| qwen_direct_select_vs_baseline_validation_n2000 | Recall@10 | 2000 | 0.314000 | 0.305000 | -0.009000 | [-0.017500, -0.000500] | 0.0418 | 0.0362397 |
| qwen_routing_vs_baseline_validation_n2000 | Accuracy@1 | 2000 | 0.071000 | 0.071000 | 0.000000 | [0.000000, 0.000000] | 1 | 1 |
| qwen_routing_vs_baseline_validation_n2000 | MRR@10 | 2000 | 0.133909 | 0.134373 | 0.000464 | [-0.000811, 0.001821] | 0.4832 | 0.487546 |
| qwen_routing_vs_baseline_validation_n2000 | NDCG@10 | 2000 | 0.175972 | 0.175087 | -0.000886 | [-0.003303, 0.001518] | 0.4728 | 0.471937 |
| qwen_routing_vs_baseline_validation_n2000 | Recall@10 | 2000 | 0.314000 | 0.308000 | -0.006000 | [-0.013000, 0.001000] | 0.1134 | 0.102327 |
