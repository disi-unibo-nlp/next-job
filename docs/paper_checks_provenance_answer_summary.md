# Paper-check Provenance Answer Summary

This note summarizes the experimental/provenance checks added for the teacher's TODOs. It is intended to accompany the final paper-check bundle and clarify which results should be used in the paper.

## Final Experimental Sources to Use

- Use the **formal BGE-M3** outputs, not the older `data_full` BGE significance/export.
- Final BGE-M3 formal per-query output:
  - `outputs/paper_checks/bge_m3_original_formal_per_query_full_test`
- Final trained ST formal per-query/significance output:
  - `outputs/paper_checks/trained_st_formal_per_query_full_test`
- Final error analysis with BGE and trained ST included:
  - `outputs/paper_checks/error_analysis_plus_bge`
- Final per-filter exact counts:
  - `outputs/paper_checks/filter_counts`
- LateOn-only full test:
  - `outputs/paper_checks/lateon_only_datafull_full_test`

## 1. BGE per-query fusion: why old value was 0.1364 while formal is 0.1397

The BGE retrieval/test split itself was not the core issue. The difference came from the fusion scorer/source used by the old `data_full` paper-check script versus the formal evaluation pipeline.

The final paper result should use the formal BGE per-query export:

- BGE-M3 pure formal test:
  - `MRR@10 = 0.0685487955868767`
  - `Recall@10 = 0.1554115117329926`
  - `NDCG@10 = 0.08904623938963163`
  - `Accuracy@1 = 0.034373149677459566`
- BGE-M3 + Markov formal fusion:
  - `alpha_markov = 0.5`
  - `alpha_embedding = 0.5`
  - `MRR@10 = 0.13967523114037544`
  - `Recall@10 = 0.307145750880364`
  - `NDCG@10 = 0.1790239140896689`
  - `Accuracy@1 = 0.07765901087600112`

Formal paired bootstrap significance:

- `MRR@10`: `0.068549 -> 0.139675`
- delta: `+0.071126`
- 95% CI: `[0.067961, 0.074330]`
- `p_boot = 0`

## 2. Trained ST fusion vs BGE-M3 paired significance

The trained ST model was evaluated with the formal per-query pipeline and compared against the formal BGE-M3 + Markov fusion output.

Trained ST final model:

- `/workspace/handoff/checkpoints/sentence_transformer/final`

Final trained ST + Markov result:

- `alpha_markov = 0.25`
- `alpha_embedding = 0.75`
- `selected_validation_MRR = 0.1571330362609201`
- `MRR@10 = 0.15470902603720696`
- `Recall@10 = 0.3398360808999969`
- `NDCG@10 = 0.1982008962677278`
- `Accuracy@1 = 0.08638474243510237`

Paired bootstrap vs formal BGE-M3 + Markov:

- `MRR@10`: `0.139675 -> 0.154709`
- delta: `+0.015034`
- 95% CI: `[0.013619, 0.016457]`
- `p_boot = 0`

This supports the claim that the trained ST retriever significantly improves over BGE-M3 + Markov fusion.

## 3. Markov and Transition KG relationship

Both Markov and Transition KG are derived from **train-only transition statistics**.

They differ in how the train transitions are used:

- Markov is the simpler first-order transition probability component used in dense/fusion evaluation.
- Transition KG is a graph-style transition scorer built from the train transition index. It uses recent history, recency decay, exact/prefix transition evidence, and prior terms.

The code evidence shows the Transition KG/Wiki components are built from train transitions:

- `scripts/jobhop_transition_wiki_eval.py`
  - builds transition wiki from train data
  - uses recent history window and recency decay
  - scores outgoing transitions from train transition counts
- `scripts/jobhop_wiki_kg_ablation.py`
  - calls `build_train_transition_index(...)`
  - computes transition ranks from train-derived transition statistics

Recommended paper wording:

> The Markov and Transition KG components are both constructed from train-only transition statistics. The Markov component uses a first-order transition score, while the Transition KG scorer uses the same source of train transitions in a graph-style ranking formulation with recent-history weighting, exact/prefix transition evidence, and prior smoothing.

## 4. KG parameters: 5 / 0.65 / 8 / 1 / 0.05

The checked code gives the following fixed configuration/defaults:

- `history_window = 5`
- `transition_recency_decay = 0.65`
- `exact_transition_weight = 8.0`
- unit rank-bonus term: `1.0 / (rank + 2)`
- `prior_weight = 0.05`
- `transition_top_k = 100`

The relevant scoring formula includes:

```text
score += exact_transition_weight * recency_weight * (probability + 1.0 / (rank + 2))
```

Recommended paper wording:

> Transition KG hyperparameters were fixed in the evaluation configuration: history window 5, recency decay 0.65, exact-transition weight 8.0, a unit rank-bonus term, and prior weight 0.05. Test results were computed once under this fixed configuration.

If the paper needs to distinguish validation tuning from fixed defaults, use conservative wording:

> These values were fixed by the development/evaluation configuration rather than tuned on the test set.

## 5. Wiki+KG 0.8 / 0.2 source

The Wiki+KG score fusion code uses:

```text
score = alpha * KG + (1 - alpha) * Wiki
```

The alpha grid includes:

```text
0.7, 0.8, 0.9, 0.95
```

The final `alpha_0.8` result therefore means:

- KG weight: `0.8`
- Wiki weight: `0.2`

Recommended paper wording:

> Wiki+KG score fusion uses a linear interpolation score, `alpha * KG + (1-alpha) * Wiki`. We use `alpha=0.8`, selected/configured from the validation alpha grid, corresponding to 0.8 KG weight and 0.2 Wiki weight.

## 6. Qwen 265 failures and metrics

For the Qwen routing validation run:

- `route_failures = 0`
- `ranking_failures = 265`
- evaluation size remains `n = 2000`

Thus the failed Qwen ranking calls were **not dropped** from metrics. They were retained in the 2,000-case evaluation and handled by fallback/deterministic candidate ranking logic.

For the Qwen direct-selection run:

- `route_failures = 0`
- `ranking_failures = 0`

Recommended paper wording:

> Qwen routing failures were retained in the evaluation set rather than removed. For failed ranking calls, the pipeline fell back to deterministic candidate ranking while preserving the same query count.

## 7. Why Qwen Candidate Recall@100 stays the same

The Qwen experiments rerank or select within the same top-100 candidate pool. Candidate generation itself is unchanged.

Therefore `candidate_recall@100` remains the same across Qwen variants:

- same candidate pool size: `100`
- same generated candidate IDs
- Qwen changes ranking/selection behavior, not the upstream candidate set

Recommended paper wording:

> Candidate Recall@100 is unchanged because Qwen operates downstream of the candidate generation stage. It reranks/selects within the same top-100 pool and does not alter the candidate set.

## 8. Per-filter exact counts

The per-filter count check has been completed under:

- `outputs/paper_checks/filter_counts`

Final counts:

- train retained trajectories: `256593`
- validation eval cases: `32123`
- test eval cases: `32089`
- train candidate vocabulary size: `2982`

Split-level exact counts:

| split | raw rows | raw persons | unknown-code rows removed | valid-code rows retained | final cases |
|---|---:|---:|---:|---:|---:|
| train | 1594827 | 284251 | 110573 | 1484254 | 256593 |
| validation | 199587 | 35531 | 13622 | 185965 | 32123 |
| test | 198877 | 35533 | 13587 | 185290 | 32089 |

For validation and test:

- dropped because target not in train candidate vocabulary: `0`

Recommended paper wording:

> After removing unknown/invalid codes and collapsing consecutive duplicates, we retain 256,593 train trajectories, 32,123 validation cases, and 32,089 test cases. All validation/test targets are covered by the train candidate vocabulary.

## 9. Final status

The experiment-side TODOs are complete:

- formal BGE-M3 per-query export and significance
- trained ST formal export and paired significance against BGE-M3
- expanded error analysis including BGE-M3 and trained ST
- per-filter exact counts
- LateOn-only full test
- provenance checks for KG/Wiki/Qwen configuration

The remaining work is mainly paper/report editing: update tables, methods text, and appendix/provenance descriptions using the formal outputs above.
