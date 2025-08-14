# Medical-Specialty-classification
# Medical-Specialty Classification (MTSamples)

This repository implements a 3-stage pipeline to classify free-text clinical notes into medical specialties using domain-adapted BERT (Bio_ClinicalBERT), a lightweight logistic-regression baseline and a feature-fusion trick (description + keywords) that improved accuracy to 89% on our experiments.

---

## Table of contents

* [Quick start](#quick-start)
* [Repo structure](#repo-structure)
* [Requirements](#requirements)
* [Dataset & preprocessing](#dataset--preprocessing)
* [Training stages (how to run)](#training-stages-how-to-run)
  * [Stage 1 — Frozen BERT + Logistic Regression](#stage-1---frozen-bert--logistic-regression)
  * [Stage 2 — End-to-end Fine-Tuning](#stage-2---end-to-end-fine-tuning)
  * [Stage 3 — Feature Fusion (description + keywords)](#stage-3---feature-fusion-description--keywords)
* [Results](#Approximate results from experiments)
* [Parameters](#Hyperparameters & tips)
---

## Quick start

1. Clone the repo and create a virtual environment:

```bash
git clone https://github.com/<your-username>/medical-specialty-classifier.git
cd medical-specialty-classifier
python -m venv .venv
source .venv/bin/activate
```

3. Run the training stages (examples below). Replace model names and paths as needed.

---

## Repo structure

```
.
├─ data/
│  └─ mt_samples.csv
├─ notebooks/
│  └─ code.ipynb
└─ README.md
```

## Requirements

Minimum tested stack:

* Python 3.9+
* PyTorch (1.12+)
* transformers (Hugging Face)
* datasets (Hugging Face)
* scikit-learn
* pandas, numpy, matplotlib, seaborn
* tqdm

## Dataset & preprocessing

**Input:** `data/raw/mt_samples.csv` (MTSamples)

**Main preprocessing steps** (implemented in `scripts/preprocess.py`):

1. Skip or attempt to fix malformed rows (`on_bad_lines='skip'` by default).
2. Drop singleton specialties (labels with only 1 example) so stratified splitting is possible.
3. Stratified splits: 80/20 train/test, then 90/10 train/val from training set.
4. Tokenization with the BERT tokenizer (`max_length=512`, `padding='max_length'`, `truncation=True`).
5. Re-fit `LabelEncoder` after filtering so label IDs are contiguous (`0..N-1`).
---
## Training stages (how to run)

### Stage 1 — Frozen BERT + Logistic Regression

**Purpose:** Quick baseline using CLS embeddings.

High-level flow:

1. Load Bio\_ClinicalBERT tokenizer & model.
2. Tokenize and run the model in eval mode to extract CLS embeddings (768-d vectors).
3. Train a scikit-learn `LogisticRegression` on the embeddings.

### Stage 2 — End-to-end Fine-Tuning

**Purpose:** Fine-tune all BERT weights on the 35-class task.

### Stage 3 — Feature Fusion (description + keywords)

**Purpose:** Concatenate a short description and extracted keywords with the transcript to disambiguate similar specialties.

**Approximate results from experiments:**

* Stage 1 (Frozen CLS + LogisticRegression): accuracy ≈ mid-70s, macro-F1 ≈ 0.34.
* Stage 2 (End-to-end fine-tune): macro-F1 improved into high-40s.
* Stage 3 (Feature fusion): accuracy ≈ **\~89%**, macro-F1 ≈ **0.60–0.64**.
---

## Hyperparameters & tips

* **Batch size:** 8 or 16 (use gradient accumulation for larger effective batch sizes).
* **Learning rate:** 2e-5 → 5e-5 typical for BERT fine-tuning.
* **Warmup:** 5–10% of total steps.
* **Weight decay:** 0.01.
* **Epochs:** 3–5 is usually sufficient.
* **Scheduler:** `get_linear_schedule_with_warmup` with `scheduler.step()` per batch.
* **Zeroing grads:** always call `optimizer.zero_grad()` before `loss.backward()` unless intentionally accumulating gradients.
---
