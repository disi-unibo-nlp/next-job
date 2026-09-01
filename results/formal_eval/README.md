# Formal Evaluation Results

This directory is organized to match the two formal result tables used in the
report.

## Normal Evaluation

`normal_eval/` contains the four original single-model evaluation results:

- `normal_eval/bge_m3/bge_m3_.json`
- `normal_eval/sentence_transformer/final_.json`
- `normal_eval/qwen3_embedding_b2/checkpoint-16038-merged_.json`
- `normal_eval/late_interaction/final_.json`

These correspond to the normal evaluation table:

- BGE-M3
- Sentence Transformer
- Qwen3 Embedding B2
- Late Interaction

## Comparison Evaluation

`comparison_eval/` contains the four comparison/fusion evaluation results:

- `comparison_eval/bge_m3/`
- `comparison_eval/sentence_transformer_trained/`
- `comparison_eval/qwen3_embedding_b2_trained/`
- `comparison_eval/lateon/`

These correspond to the comparison evaluation table:

- BGE-M3
- Sentence Transformer trained
- Qwen3 Embedding B2 trained
- LateOn

Each comparison directory keeps its `final_test_results`, `validation_sweep`,
`selected_alpha`, `run_config`, runtime, dataset, and environment metadata.

## Summary Table

`formal_eval_summary.csv` is a compact table with the main metrics used for
reporting:

- Accuracy@1
- MRR@10
- NDCG@10
- Recall@10
- MAP@10

The first four rows are `normal_eval`, and the last four rows are
`comparison_eval`.
