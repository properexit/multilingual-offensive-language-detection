# Offensive Language Detection (OffensEval 2020)

This project implements offensive language detection using the OffensEval 2020 (SemEval Task 12) dataset.
It supports English (Tasks A, B, C) and Arabic (Task A), and demonstrates cross-lingual transfer learning using multilingual transformers.

---

## Tasks Supported

| Language | Task | Description |
|--------|------|-------------|
| English | A | Offensive language identification |
| English | B | Offense type (Targeted vs Untargeted) |
| English | C | Offense target (Individual / Group / Other) |
| Arabic | A | Offensive language identification |

---

## Models Used

**English**
- BERT-mini  
  `google/bert_uncased_L-2_H-128_A-2`

**Arabic / Multilingual**
- XLM-R  
  `xlm-roberta-base`

---

## Project Structure

```
cv_offensive/
│
├── main.py                 # Single entry point
├── config/                 # Experiment configs
├── datasets/               # Language-specific loaders
├── models/                 # Baseline & transformer models
├── training/               # Training logic
├── utils/                  # Device, metrics, seed
├── data/                   # (ignored) raw datasets
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create Environment

```
python -m venv venv-offensive
source venv-offensive/bin/activate
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

## How to Run

All experiments are executed via `main.py`.

### English – Task A (Offensive Detection)

```
python main.py --lang english --task A --config config/english.yaml
```

### English – Task B (Offense Type)

```
python main.py --lang english --task B --config config/english.yaml
```

### English – Task C (Offense Target)

```
python main.py --lang english --task C --config config/english.yaml
```

### Arabic – Task A (Zero-shot)

Evaluate XLM-R on Arabic without Arabic training.

```
python main.py --lang arabic --task A --mode zero-shot --config config/arabic.yaml
```

### Arabic – Task A (Few-shot Transfer Learning)

English to Arabic transfer with 500 Arabic samples.

```
python main.py --lang arabic --task A --mode few-shot --k 500 --config config/arabic.yaml
```

---

## Device Selection

By default, the system auto-selects the best available device:
- CUDA (GPU)
- MPS (Apple Silicon)
- CPU

To force a device manually:

```
--device cpu
--device cuda
--device mps
```

Example:

```
python main.py --lang english --task A --device cpu
```

---

## Final Results (Macro F1)

| Language | Task | Setting | Macro F1 |
|--------|------|---------|----------|
| English | A | Supervised | 0.91 |
| Arabic | A | Zero-shot | 0.45 |
| Arabic | A | Few-shot (500) + transfer | 0.63 |

---

## Notes

- Arabic Tasks B and C are not supported due to lack of labeled data.
- Few-shot Arabic experiments use English pretraining for transfer learning.
- Raw datasets are excluded from the repository (see `.gitignore`).
