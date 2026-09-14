# Formal Preprocessing Filter Counts

Counts are computed with the same formal preprocessing utilities used by the evaluation scripts.

## Key Results

### train

- raw rows: `1594827`
- raw persons: `284251`
- Unknown-code rows removed: `110573`
- valid-code rows retained: `1484254`
- len>=2 trajectories after consecutive duplicate collapse: `256593`
- final train trajectories: `256593`
- train candidate vocabulary size: `2982`

### validation

- raw rows: `199587`
- raw persons: `35531`
- Unknown-code rows removed: `13622`
- valid-code rows retained: `185965`
- len>=2 trajectories after consecutive duplicate collapse: `32123`
- eval cases after len filter: `32123`
- eval cases retained with target in train vocab: `32123`
- dropped because target not in train vocab: `0`
- final eval cases: `32123`

### test

- raw rows: `198877`
- raw persons: `35533`
- Unknown-code rows removed: `13587`
- valid-code rows retained: `185290`
- len>=2 trajectories after consecutive duplicate collapse: `32089`
- eval cases after len filter: `32089`
- eval cases retained with target in train vocab: `32089`
- dropped because target not in train vocab: `0`
- final eval cases: `32089`

## Final Counts

- train retained trajectories: `256593`
- validation eval cases: `32123`
- test eval cases: `32089`
- train candidate vocabulary size: `2982`

## CSV Columns

`split`, `raw_rows`, `raw_persons`, `unknown_code_rows`, `valid_code_rows_after_unknown_invalid_removed`, `dropped_rows_unknown_or_invalid`, `persons_len_ge_2_after_collapse`, `final_train_trajectories`, `train_candidate_vocab_size`, `eval_cases_after_len_filter`, `eval_cases_target_in_train_vocab`, `dropped_eval_cases_target_not_in_train_vocab`, `final_eval_cases`

## Files

- `filter_counts_by_split.csv`
- `filter_counts_stages.csv`
- `filter_counts_summary.json`
- `filter_counts_summary.md`
