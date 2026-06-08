#!/bin/bash
##############################################################################
# Damage-TriageFormer: 30-epoch training on 2x A100 (40GB) with torchrun/DDP.
#
# Effective batch = 32 = batch_size 2 x 2 GPUs x accumulation 8.
# Edit the SBATCH directives, module loads, and conda env to match your cluster.
#
# Submit: sbatch scripts/train_2gpu.sbatch
##############################################################################

#SBATCH --job-name=damage_triageformer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Offline mode: download the DINOv3 backbone once on a login node first.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Point these at your dataset and output locations.
export DINOV3_DATA_ROOT=./data
export DINOV3_RUNS_ROOT=./runs
mkdir -p logs "${DINOV3_RUNS_ROOT}"

# --- activate your environment (edit to match your cluster) ---
# module load Anaconda3
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate dinov3_env

torchrun --nproc_per_node=2 --standalone main.py \
    --name damage_triageformer_v11 \
    --epochs 30 \
    --batch-size 2 \
    --num-workers 8 \
    --unfreeze-blocks 4 \
    --lr 5e-5 \
    --label-smoothing 0.1 \
    --use-gated-head \
    --gate-weight 1.0 \
    --leaf-weight 2.0 \
    --use-aux-severity \
    --aux-severity-weight 0.5 \
    --aux-severity-targets 0.0,0.3,0.7,0.5,1.0 \
    --accumulation-steps 8 \
    --no-focal-loss \
    --no-weighted-sampler \
    --la-tau 1.0 \
    --use-ema \
    --ema-decay 0.9995 \
    --use-sfp \
    --sfp-stride 8 \
    --no-multiscale \
    --no-attention-pool \
    --stratified-splits ${DINOV3_DATA_ROOT}/tiles_1024/stratified_splits.json \
    --val-split-key val

echo "Training finished at $(date)"
