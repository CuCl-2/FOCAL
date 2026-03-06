# FOCAL: FOCAL: Fine-Grained Optimal-Transport-Driven Contrastive Alignment of Language and ECGs with Waveform Enhancement

## Overview

FOCAL is a self-supervised pre-training framework that learns **fine-grained alignment** between 12-lead ECG signals and clinical report tags. Instead of treating each ECG-report pair as a single global match, FOCAL identifies which ECG segments are most relevant to each specific clinical finding (e.g., *atrial fibrillation*, *myocardial infarction*).

### Key Contributions

**1. Spatial-Temporal ECG Encoding**
Each 12-lead ECG is divided into `L × N_time` non-overlapping patches (default: 12 leads × 10 time windows = 120 patches). Learnable lead embeddings and temporal embeddings preserve spatial and temporal coordinates, enabling the model to distinguish both *which lead* and *when* a pathological event occurs.

**2. Tag-Level Report Encoding**
Clinical reports are split by comma into individual diagnostic tags. Each tag is encoded independently with BioClinicalBERT, enabling fine-grained text representations at the semantic level rather than the document level.

**3. Semi-Unbalanced Optimal Transport (Semi-UOT)**
Tag-patch alignment is formulated as a Semi-UOT problem:
- **Tag side** (balanced, strict marginal): every clinical tag must be fully grounded to the ECG waveform.
- **Patch side** (unbalanced, KL penalty): background or uninformative ECG patches receive near-zero transport mass, preventing erroneous alignment of normal baselines to pathological tags.

The log-domain Sinkhorn algorithm makes this module fully differentiable and trainable end-to-end.

**4. Semantic-Guided Soft Sigmoid Loss**
Instead of treating all non-paired tags as hard negatives (SigLIP), a text-to-text semantic similarity matrix `S` is used as soft training labels, eliminating the false-negative problem when the same pathology appears in multiple reports within a batch.

**5. Coarse-to-Fine Training**
- **Coarse stage**: train on original ECG-report pairs (10 epochs).
- **LLM augmentation**: query Qwen3-8B for latent waveform features per diagnosis, then filter candidates using the coarse model (cosine similarity threshold 0.95).
- **Fine stage**: continue training on LLM-augmented reports (3 epochs).

---

## Repository Structure

```
FOCAL/
├── pretrain/
│   ├── main.py              # DDP pre-training entry point
│   ├── config.yaml          # Pre-training hyperparameters
│   ├── launch.sh            # torchrun launch script
│   └── preprocess.ipynb     # MIMIC-ECG preprocessing
├── utils/
│   ├── vit1d.py             # SpatialTemporalViT encoder
│   ├── utils_builder.py     # FOCAL model (ECG encoder + text encoder + Semi-UOT)
│   ├── utils_loss.py        # focal_loss (semantic-guided soft sigmoid)
│   ├── utils_trainer.py     # FOCALTrainer (DDP, variable-length tag handling)
│   ├── utils_dataset.py     # MIMIC-ECG dataset loader
│   └── zeroshot_val.py      # Zero-shot evaluation utilities
├── zeroshot/
│   ├── test_zeroshot.py     # Zero-shot test script
│   ├── zeroshot_config.yaml # Paths and model config for zero-shot testing
│   └── CKEPE_prompt.json    # Class-name → text prompt mapping
└── finetune/
    ├── main_single.py       # Linear probing / fine-tuning on downstream datasets
    ├── finetune_dataset.py  # PTB-XL / CPSC2018 / CSN dataset loaders
    └── models/
        └── vit1d.py         # SpatialTemporalViT (mirrors utils/vit1d.py)
```

---

## Installation

```bash
pip install torch torchvision transformers einops wandb wfdb scipy scikit-learn
```

> The Semi-UOT solver is implemented natively in PyTorch (log-domain Sinkhorn) — no external optimal transport library is required.

---

## Data Preparation

### Pre-training: MIMIC-IV-ECG

Download the [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) dataset and preprocess it into `.npy` arrays and `.csv` metadata following `pretrain/preprocess.ipynb`.

Expected layout after preprocessing:
```
your_data_path/
├── mimic_ecg_train.npy    # shape (N_train, 12, 5000), stored × 1000 as int16
├── mimic_ecg_val.npy
├── train.csv              # columns: study_id, total_report
└── val.csv
```

The `total_report` column should contain comma-separated clinical tags, e.g.:
```
"sinus rhythm, left axis deviation, first degree av block, st depression"
```

### Downstream Datasets

| Dataset | Reference | Task |
|---------|-----------|------|
| **PTB-XL** | [Wagner et al., 2020](https://physionet.org/content/ptb-xl/1.0.3/) | Super-class / Sub-class / Form / Rhythm |
| **CPSC2018** | [Liu et al., 2018](http://2018.icbeb.org/Challenge.html) | Multi-label arrhythmia |
| **CSN** | [Zheng et al., 2022](https://physionet.org/content/ecg-arrhythmia/1.0.0/) | Multi-label arrhythmia |

Follow the benchmark setup from [Liu et al., 2024](https://arxiv.org/abs/2406.06xxx). Place datasets under `your_path/downstream/` and data splits under `your_path/FOCAL/finetune/data_split/`.

---

## Pre-Training

### 1. Edit `pretrain/config.yaml`

Fill in the path fields:
```yaml
dataset:
  data_path: '/path/to/preprocessed/mimic'

zeroshot:
  prompt_dict:    '/path/to/FOCAL/zeroshot/CKEPE_prompt.json'
  meta_data_path: '/path/to/downstream'
  meta_split_path: '/path/to/FOCAL/finetune/data_split'
```

### 2. Coarse Stage (10 epochs)

```bash
export OMP_NUM_THREADS=8
cd pretrain
torchrun --nnodes=1 --nproc_per_node=8 \
         --rdzv_id=101 --rdzv_endpoint=localhost:29502 \
         main.py
```

Checkpoints are saved to `../checkpoints/`. Best zero-shot checkpoint is saved as `focal_vit_tiny_best_ckpt.pth`.

### 3. LLM Feature Augmentation

For each report, query Qwen3-8B with:
> *"What waveform features are most likely to be present in electrocardiograms with these symptoms: {tags}?"*

Score each candidate feature against the actual ECG using the coarse FOCAL model. Retain features with cosine similarity ≥ 0.95 and append them to the report's tag list.

### 4. Fine Stage (3 epochs)

Set `max_epochs: 3` in `config.yaml`, load the coarse checkpoint, and re-run pre-training with the augmented dataset.

---

## Zero-Shot Classification

```bash
cd zeroshot
# 1. Edit zeroshot_config.yaml: fill in your paths
# 2. Edit test_zeroshot.py:  set ckpt_path
python test_zeroshot.py
```

Macro AUROC is reported for each dataset split.

---

## Linear Probing

```bash
cd finetune

# PTB-XL super-class, frozen backbone, 100% training data
python main_single.py \
    --dataset ptbxl_super_class \
    --backbone vit_tiny \
    --ratio 100 \
    --name LinearProbing \
    --pretrain_path ../checkpoints/focal_vit_tiny_final_encoder.pth

# 10% data
python main_single.py \
    --dataset ptbxl_super_class \
    --backbone vit_tiny \
    --ratio 10 \
    --name LinearProbing \
    --pretrain_path ../checkpoints/focal_vit_tiny_final_encoder.pth
```

Available `--backbone` options: `vit_tiny`, `vit_small`, `vit_middle`, `vit_base`.
Available `--ratio` options: `1`, `10`, `100`.

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| ECG encoder | `vit_tiny` | SpatialTemporalViT (width=192, depth=12, heads=3) |
| Total patches | 120 | 12 leads × 10 time windows (500 samples/patch at 500 Hz) |
| Text encoder | BioClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` |
| Projection dim | 256 | Shared ECG-text latent space |
| Batch size | 100 | Per-GPU |
| Optimizer | AdamW | β=(0.9, 0.999) |
| Learning rate | 2×10⁻⁵ | Cosine decay with 10% linear warmup |
| Weight decay | 1×10⁻⁴ | |
| Temperature | log(10) | Learnable, initialized to log(10) |
| UOT ε | 0.1 | Entropy regularization strength |
| UOT τ | 1.0 | KL penalty on ECG-patch marginal |
| Sinkhorn iters | 30 | Log-domain for fp16 stability |
| Frozen BERT layers | 9 | First 9 layers frozen during pre-training |
| Coarse epochs | 10 | Training on original reports |
| Fine epochs | 3 | Training on LLM-augmented reports |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{focal2025,
  title   = {FOCAL: Fine-Grained Optimal Transport Contrastive Alignment for ECG-Report Pairs},
  author  = {},
  journal = {arXiv},
  year    = {2025},
}
```
