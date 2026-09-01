# next-job

## Docker setup

Build the project image on the cluster before submitting jobs:

```bash
docker build -f build/Dockerfile -t jobhop-v2:latest .
```

Submit jobs with the GPU resource requested on the command line, for example:

```bash
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 slurm/bge_m3_dense_eval.slurm
```

The SLURM scripts run the evaluation inside Docker through `run_docker.sh`. They do not activate Conda on the host.

## Main Scripts

- `scripts/dense_eval.py`: dense retriever evaluation.
- `scripts/lateon_eval.py`: LateOn/late-interaction evaluation.
- `scripts/jobhop_qwen_make_entity_pages.py`: frozen Qwen entity Wiki generation.
- `scripts/jobhop_qwen_wiki_direct_select.py`: JobHop Wiki candidate generation with optional Qwen direct selection.

## Results

Formal evaluation outputs are organized under `results/formal_eval/`.
Wiki experiment summaries are organized under `results/wiki_eval/`.
