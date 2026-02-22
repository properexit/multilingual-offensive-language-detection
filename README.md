# Multilingual Offensive Language Detection (OffensEval 2020)

This project implements offensive language detection using the OffensEval 2020 dataset.

Supported:
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

Create virtual environment

python3 -m venv venv
source venv/bin/activate

Install dependencies

pip install -r requirements.txt

---

## 2. Dataset

Place data in the following structure:

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

Arabic dataset: OffensEval 2020 Task 12 (Arabic Subtask A).

---

## 3. Running Baselines

Majority + TF-IDF baseline (English Task A)

python -m training.train_baseline

---

## 4. Transformer Training

All experiments run via main.py.

### English Task A

python main.py --lang english --task A

With stronger config:

python main.py --lang english --task A --config config/english.yaml

---

### English Task B

python main.py --lang english --task B --config config/english.yaml

---

### English Task C

python main.py --lang english --task C --config config/english.yaml

---

## 5. Multi-task Learning (Task A + B)

python main.py --multitask

This trains a shared encoder with two classification heads.

---

## 6. Arabic Experiments

### Zero-shot (English-trained XLM-R → Arabic)

python main.py --lang arabic --task A --mode zero-shot

---

### Few-shot Transfer

python main.py --lang arabic --task A --mode few-shot --k 500

This:
1. Pretrains on English
2. Fine-tunes on k Arabic samples

---

## 7. Parameter-Efficient Fine-Tuning (PEFT)

### Freeze encoder

python main.py --lang english --task A --peft freeze

### LoRA

python main.py --lang english --task A --peft lora

Arabic example:

python main.py --lang arabic --task A --mode few-shot --k 500 --peft lora

---

## 8. Configuration

Hyperparameters are controlled via YAML files.

Example:

config/base.yaml
config/english.yaml
config/arabic.yaml

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

Force device manually:

python main.py --lang english --task A --device cpu
