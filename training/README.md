# Training Code

This directory contains the training handoff code used for the learned retrievers.

Subdirectories:

- `sentence_transformer/`: task-adapted Sentence Transformer training.
- `late_interaction/`: LateOn/late-interaction training and evaluation code.
- `bge_m3_flagembedding/`: BGE-M3/FlagEmbedding training handoff.
- `qwen3_embedding/`: Qwen3 embedding training handoff.
- `shared/`: shared training/export utilities.

Large trained checkpoints are not included in this GitHub-ready project. They can be regenerated from the training code or restored from external storage.

Why checkpoints are excluded:

- They are large binary artifacts.
- They are environment-dependent.
- The final evaluation outputs and configs are already included under `results/`.

The paper's formal test metrics are reproducible from the included code plus the corresponding data and checkpoints, but the repository itself is intentionally kept lightweight.

