# Damage-TriageFormer

> **Revision note.** The original entry point (`main.py`) implements the tile-mode
> model. The revised manuscript's per-building crop mode is provided separately
> in `scripts/train_crop.py`. The original public checkpoint was trained
> on a superseded tile-level split; do not present its 0.619 test macro F1 as the
> revised paper result. The corrected photo-atomic tile-mode result is 0.579, and
> the revised crop-mode result is 0.616.

A foundation-model framework for decision-relevant building damage typology from
single post-event imagery. Given a post-event RGB tile and building instance
masks (footprints), the model assigns each building one of five damage-typology
classes:

| Class | Name |
|------:|------|
| 0 | Undamaged |
| 1 | Partial Roof Damage |
| 2 | Total Roof Damage |
| 3 | Partial Structural Damage |
| 4 | Total Structural Collapse |

The tile-mode model adapts a **DINOv3 ViT-L/16** backbone with a **Simple Feature Pyramid**
for higher-resolution instance pooling, a **two-stage gated damage head**
(any-damage gate → 4-way damaged-class leaf), long-tailed **logit adjustment**,
and an **auxiliary severity-regression** objective. It is trained and evaluated
on **DamageTriage-Bench** (Hurricane Michael 2018, Hurricane Helene 2024, and the
2025 Los Angeles wildfire complex).

This repository contains tile-mode training/inference code and crop-mode
training/evaluation code. The full analysis and figure-reproduction scripts are
maintained in the paper's revision archive.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA-capable GPU and PyTorch built for your CUDA version. The DINOv3
backbone (`facebook/dinov3-vitl16-pretrain-lvd1689m`) is fetched from Hugging Face
on first use; on offline clusters, download it once on a login node and set
`HF_HUB_OFFLINE=1`.

## Data layout

Download **DamageTriage-Bench** (see *Dataset* below) and arrange it so that the
following paths exist under your data root:

```
<DATA_ROOT>/tiles_1024/images/            # 1024x1024 post-event RGB tiles (PNG)
<DATA_ROOT>/tiles_1024/damage/            # matching RGB damage-typology masks (PNG)
<DATA_ROOT>/tiles_1024/photo_splits.json        # revised photo-atomic manifest
<DATA_ROOT>/tiles_1024/stratified_splits.json   # superseded tile-level manifest
```

Point the code at it with an environment variable (default is `./data`):

```bash
export DINOV3_DATA_ROOT=/path/to/your/data
export DINOV3_RUNS_ROOT=/path/to/your/runs   # checkpoints/logs (default ./runs)
```

## Training

**Two-GPU DDP (the reported recipe: 30 epochs, effective batch 32):**

```bash
sbatch scripts/train_2gpu.sbatch     # Slurm; edit directives for your cluster
```

or directly with `torchrun`:

```bash
torchrun --nproc_per_node=2 --standalone main.py \
    --name damage_triageformer_v11 --epochs 30 --batch-size 2 \
    --unfreeze-blocks 4 --lr 5e-5 --label-smoothing 0.1 \
    --use-gated-head --gate-weight 1.0 --leaf-weight 2.0 \
    --use-aux-severity --aux-severity-weight 0.5 \
    --aux-severity-targets 0.0,0.3,0.7,0.5,1.0 \
    --accumulation-steps 8 --no-focal-loss --no-weighted-sampler \
    --la-tau 1.0 --use-ema --ema-decay 0.9995 \
    --use-sfp --sfp-stride 8 --no-multiscale --no-attention-pool \
    --stratified-splits $DINOV3_DATA_ROOT/tiles_1024/photo_splits.json \
    --val-split-key val
```

**Single GPU** (smaller effective batch; raise `--accumulation-steps` to compensate):

```bash
python main.py --name damage_triageformer_v11 --epochs 30 --batch-size 2 \
    --unfreeze-blocks 4 --use-gated-head --use-aux-severity --use-sfp --sfp-stride 8 \
    --la-tau 1.0 --use-ema --no-focal-loss --no-weighted-sampler \
    --stratified-splits $DINOV3_DATA_ROOT/tiles_1024/photo_splits.json
```

The best checkpoint (by validation macro-F1 in the footprint-conditioned setting)
is written to `$DINOV3_RUNS_ROOT/<name>/best_model.pth`.

### Primary crop-mode configuration

Build the connected-component instance index once, extract the per-building
crops, and train the gated DINOv3 configuration:

```bash
python scripts/build_instance_index.py \
    --splits photo_splits.json \
    --out instance_index

python scripts/train_crop.py --prepare \
    --crops /path/to/crops \
    --index-dir instance_index

python scripts/train_crop.py \
    --arch dinov3_vitl16 --gated \
    --crops /path/to/crops \
    --index-dir instance_index \
    --splits photo_splits.json \
    --epochs 20 --batch 48 --lr 3e-5 \
    --name crop_dtf
```

The same script produces the crop baselines with `--arch resnet50`,
`--arch vit_b_16`, or `--arch dinov3_vitl16` without `--gated`.

## Results and protocol scope

- Revised photo-atomic split, tile-mode full recipe: macro F1 **0.575**
  (validation) / **0.579** (test).
- Revised photo-atomic split, crop-mode primary configuration: macro F1 **0.646**
  (validation) / **0.616** (test).
- Superseded tile-level split, archived tile checkpoint: macro F1 **0.624**
  (validation) / **0.619** (test). This is retained only for historical
  reproducibility and is not the revised manuscript protocol.

All configurations are single-seed. The photo-atomic split keeps tiles derived
from one source photo in one fold; it does not reconstruct buildings clipped at
tile boundaries or rule out duplicate buildings across overlapping photos.

## Repository layout

```
main.py                 # training/inference entry point (DDP-aware)
src/config.py           # central config; paths via DINOV3_DATA_ROOT / DINOV3_RUNS_ROOT
src/model/              # DINOv3 backbone, SFP, gated/severity heads, Model
src/data/               # dataset, dataloaders, samplers, preprocessing
src/training/           # train/validate loops and losses
src/utils/              # class weights, visualization, seeding
scripts/train_2gpu.sbatch
scripts/build_instance_index.py
scripts/train_crop.py
photo_splits.json
```

## Dataset

DamageTriage-Bench (1024×1024 tiles, building-instance damage-typology masks,
photo-atomic train/val/test splits) is released separately under **CC-BY-NC-4.0**:
<https://huggingface.co/datasets/Ymx1025/DamageTriage-Bench>.

## License

Code is released under the **MIT License** (see `LICENSE`). The DamageTriage-Bench
dataset is released under **CC-BY-NC-4.0** and is subject to the redistribution
terms of the underlying NOAA Emergency Response Imagery and source building-footprint
layers.

## Citation

```bibtex
@misc{xiao2026damagetriageformerfoundationmodelframeworktypologybased,
      title={Damage-TriageFormer: A Foundation-Model Framework for Typology-Based Building Damage Assessment from Mono-Temporal Imagery},
      author={Yiming Xiao and Yu-Hsuan Ho and Sanjay Thasma and Junwei Ma and Ali Mostafavi},
      year={2026},
      eprint={2606.12248},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.12248},
}
```
