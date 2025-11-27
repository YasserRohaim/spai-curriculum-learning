# SPAI – Curriculum Fine-Tuning & Inference (Project README)

This repository contains our **fine-tuning, curriculum scheduling, multi-epoch inference, and evaluation utilities** built on top of the original SPAI detector.

> **Note:** Base model weights for the original SPAI can be obtained from the **original repository**. Use those as initialization for fine-tuning here.

---

## 📦 Environment & Setup

We use Python 3.11 and CUDA-enabled PyTorch.

```bash
conda create -n spai python=3.11
conda activate spai
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

If you plan to train, install [NVIDIA APEX](https://github.com/NVIDIA/apex) (optional but recommended for AMP).

Place the original SPAI weights under `./weights/` (download from the original repo):

```
weights/
└─ mfm_pretrain_vit_base.pth
```

---

## 🗂️ Datasets

We train/evaluate from CSVs with at least:
- `image` (path relative to a root directory),
- `class` (0 = real, 1 = generated),
- `split` (e.g., `train`, `val`, `test`).

Examples used in our scripts:
- **SD3**: CSV `stablediffusion3-sd3-v3.csv` with root `stablediffusion3-sid/versions/3`
- **ITW-SM**: CSV `output/itwsm.csv` with root `ITW-SM/`

---

## 🎯 Fine-Tuning with Curriculum

We enable a **matched-vs-synthetic curriculum** during training. We keep the matched fraction at **0.0 for the first 2 epochs** and linearly ramp it to **0.15**.

The following config overrides work with our fork:
- `DATA.CURRICULUM.ENABLED` (bool)
- `DATA.CURRICULUM.START_MATCHED_FRACTION` (float)
- `DATA.CURRICULUM.END_MATCHED_FRACTION` (float)
- `DATA.CURRICULUM.WARMUP_EPOCHS_ZERO` (int, optional; if absent, skip it)
- `DATA.CURRICULUM.RAMP_EPOCHS` (int)

### Example: fine-tune with curriculum (SD3 CSV)

```bash
python -m spai train \
  --cfg ./configs/spai.yaml \
  --batch-size 72 \
  --pretrained ./weights/mfm_pretrain_vit_base.pth \
  --output ./output/train \
  --data-path ./stablediffusion3-sd3-v3.csv \
  --csv-root-dir ./stablediffusion3-sid/versions/3 \
  --tag spai_curriculum \
  --amp-opt-level O2 \
  --data-workers 8 \
  --save-all \
  --opt DATA.CURRICULUM.ENABLED true \
  --opt DATA.CURRICULUM.START_MATCHED_FRACTION 0.0 \
  --opt DATA.CURRICULUM.END_MATCHED_FRACTION 0.15 \
  --opt DATA.CURRICULUM.WARMUP_EPOCHS_ZERO 2 \
  --opt DATA.CURRICULUM.RAMP_EPOCHS 8
```

> If your configuration does **not** support `WARMUP_EPOCHS_ZERO`, emulate a 2-epoch flat start by beginning the ramp after epoch 2.

---

## 🔎 Inference (single run)

Run inference on a CSV (valid for **val/test** splits). This writes an output CSV with predictions to `--output`.

```bash
python -m spai infer \
  --cfg ./configs/spai.yaml \
  --model ./output/train/finetune/spai/ckpt_epoch_X.pth \
  --input ./stablediffusion3-sd3-v3.csv \
  --input-csv-root-dir ./stablediffusion3-sid/versions/3 \
  --split val \
  --output ./output/sd3_val_infer
```

**ITW-SM example:**
```bash
python -m spai infer \
  --cfg ./configs/spai.yaml \
  --model ./output/train/finetune/spai/ckpt_epoch_X.pth \
  --input ./output/itwsm.csv \
  --input-csv-root-dir ./ITW-SM \
  --split val \
  --output ./output/itwsm_val_infer
```

---

## 🔁 Inference Sweep over Epochs

We provide a POSIX-`sh` compatible script to iterate `ckpt_epoch_0..10.pth` and write per-epoch CSVs.

Create `scripts/infer_sweep.sh`:

```sh
#!/bin/sh
CSV_PATH="stablediffusion3-sd3-v3.csv"
CSV_ROOT="stablediffusion3-sid/versions/3"
CKPT_DIR="output/sd3_v3/finetune/curriculum_2"
OUT_BASE="output"
RUN_CMD="python -m spai infer"
DATA_FLAGS="--input \"$CSV_PATH\" --input-csv-root-dir \"$CSV_ROOT\" --split val"

i=0
while [ "$i" -le 10 ]; do
  CKPT="$CKPT_DIR/ckpt_epoch_${i}.pth"
  OUTDIR="$OUT_BASE/curriculum_epoch$i"
  [ -f "$CKPT" ] || { echo "[warn] $CKPT missing"; i=$((i+1)); continue; }
  mkdir -p "$OUTDIR"
  echo "[info] epoch $i -> $OUTDIR"
  eval $RUN_CMD --cfg configs/spai.yaml --model \"$CKPT\" $DATA_FLAGS --output \"$OUTDIR\"
  i=$((i+1))
done
echo "[done] processed epochs 0..10"
```

Make executable and run:
```bash
chmod +x scripts/infer_sweep.sh
sh scripts/infer_sweep.sh
```

To run on **ITW-SM**, change in the script:
```
CSV_PATH="output/itwsm.csv"
CSV_ROOT="ITW-SM/"
```

---

## 📊 Evaluation & Plots

We provide two utilities:

### 1) Multi-epoch evaluation

Evaluates epoch directories `output/curriculum_epoch{0..10}/stablediffusion3-sd3-v3.csv`, computing:
- **ACC@0.5**, **Oracle-ACC** (best threshold), **AUC**, **TPR**, **TNR** (standard rates, not at oracle),
- Subsets: **real+matched**, **real+synthetic**,
- Optional **TOTAL** (e.g., full ≈15k) if present in the file,
- Saves a summary CSV and plots under `metric_output/`.

```bash
python evaluate_epochs.py \
  --root output \
  --basename curriculum_epoch \
  --start 0 --end 10 \
  --pred-csv-name stablediffusion3-sd3-v3.csv \
  --out metric_output
```

### 2) Single-CSV evaluation

If you have a single predictions file (e.g., `output/stablediffusion3-sd3-v3.csv`):

```bash
python evaluate_single.py \
  --csv output/stablediffusion3-sd3-v3.csv \
  --out metric_output_single
```

Both scripts expect the prediction CSV to include ground-truth (`class`) and a score/probability column (written by `spai infer` under your `--tag`, default `spai`).

---



## 🙏 Acknowledgments

- Original SPAI paper & repository authors for the base model and training code.
- Our repo extends SPAI with curriculum fine-tuning utilities, sweep inference, and evaluation tooling tailored to our datasets.

---

## 📄 License

Our additions follow the same license terms as the original project (Apache-2.0). Third-party datasets and dependencies retain their respective licenses. Review and comply with their terms before use.
