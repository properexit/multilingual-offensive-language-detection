# Multilingual Offensive Language Detection (OffensEval 2020)

This project implements offensive language detection using the OffensEval 2020 dataset.

## Supported
- English: Task A, B, C
- Arabic: Task A
- Baselines (Majority, TF-IDF + Logistic Regression)
- Transformer models (BERT / XLM-R)
- Zero-shot transfer (English → Arabic)
- Few-shot transfer
- Multi-task learning (Task A + B)
- Parameter-efficient fine-tuning (LoRA / Freeze)

---

## 1. Setup

### Clone the repository
```bash
git clone https://github.com/properexit/multilingual-offensive-language-detection.git
cd multilingual-offensive-language-detection
```

### Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## 2. Dataset

Place data in the following structure:

```
data/raw/
    english/
        test_a_tweets.tsv
        test_a_labels.csv
        test_b_tweets.tsv
        test_b_labels.csv
        test_c_tweets.tsv
        test_c_labels.csv

    arabic/
        offenseval-ar-training-v1/
            offenseval-ar-training-v1.tsv
```

Arabic dataset: OffensEval 2020 Task 12 (Arabic Subtask A).

---

## 3. Running Baselines

### Majority + TF-IDF baseline (English Task A)
```bash
python -m training.train_baseline
```

---

## 4. Transformer Training

All experiments run via main.py.

### English Task A
```bash
python main.py --lang english --task A
```

With stronger config:
```bash
python main.py --lang english --task A --config config/english.yaml
```

### English Task B
```bash
python main.py --lang english --task B --config config/english.yaml
```

### English Task C
```bash
python main.py --lang english --task C --config config/english.yaml
```

---

## 5. Multi-task Learning (Task A + B)

```bash
python main.py --multitask
```

This trains a shared encoder with two classification heads.

---

## 6. Arabic Experiments

### Zero-shot (English-trained XLM-R → Arabic)
```bash
python main.py --lang arabic --task A --mode zero-shot
```

### Few-shot Transfer
```bash
python main.py --lang arabic --task A --mode few-shot --k 500
```

This:
1. Pretrains on English  
2. Fine-tunes on k Arabic samples  

---

## 7. Parameter-Efficient Fine-Tuning (PEFT)

### Freeze encoder
```bash
python main.py --lang english --task A --peft freeze
```

### LoRA
```bash
python main.py --lang english --task A --peft lora
```

### Arabic example
```bash
python main.py --lang arabic --task A --mode few-shot --k 500 --peft lora
```

---

## 8. Configuration

Hyperparameters are controlled via YAML files.

Example:
- config/base.yaml
- config/english.yaml
- config/arabic.yaml

Typical parameters:
- batch_size
- epochs
- learning_rate
- max_length
- class_weighted

---

## 9. Device Selection

Auto-detects:
- CUDA (if available)
- Apple MPS
- CPU

### Force device manually
```bash
python main.py --lang english --task A --device cpu
```
