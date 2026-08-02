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
