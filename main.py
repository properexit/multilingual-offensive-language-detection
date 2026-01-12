import argparse
import yaml

from utils.device import get_device
from training.train_transformer import train_transformer

from datasets.english.loaders import (
    load_task_a_english,
    load_task_b_english,
    load_task_c_english
)
from datasets.arabic.loaders import load_task_a_arabic


MODEL_MAP = {
    "english": "google/bert_uncased_L-2_H-128_A-2",
    "arabic": "xlm-roberta-base"
}


def main():
    parser = argparse.ArgumentParser(
        description="OffensEval 2020 – Multilingual Offensive Language Detection"
    )

    parser.add_argument("--lang", required=True, choices=["english", "arabic"])
    parser.add_argument("--task", required=True, choices=["A", "B", "C"])
    parser.add_argument(
        "--mode",
        default="finetune",
        choices=["finetune", "zero-shot", "few-shot"]
    )
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(
        prefer_gpu=(args.device == "auto"),
        force_device=None if args.device == "auto" else args.device
    )

    print(f"Device: {device}")
    print(f"Language: {args.lang} | Task: {args.task} | Mode: {args.mode}")

    # ============================
    # ENGLISH
    # ============================
    if args.lang == "english":
        loader = {
            "A": load_task_a_english,
            "B": load_task_b_english,
            "C": load_task_c_english,
        }[args.task]

        train_df, dev_df = loader()
        num_labels = 2 if args.task != "C" else 3

        train_transformer(
            model_name=MODEL_MAP["english"],
            train_df=train_df,
            dev_df=dev_df,
            num_labels=num_labels,
            language="english",
            mode=args.mode,
            few_shot_k=args.k,
            config=config,
            device=device
        )
        return

    # ============================
    # ARABIC
    # ============================
    if args.task != "A":
        raise ValueError("Arabic supports Task A only")

    # ---- ZERO-SHOT ----
    if args.mode == "zero-shot":
        ar_train, ar_dev = load_task_a_arabic()

        train_transformer(
            model_name=MODEL_MAP["arabic"],
            train_df=ar_train,
            dev_df=ar_dev,
            num_labels=2,
            language="arabic",
            mode="zero-shot",
            few_shot_k=None,
            config=config,
            device=device
        )
        return

    # ---- FEW-SHOT WITH TRANSFER ----
    print("Stage 1: English pretraining for transfer...")
    en_train, en_dev = load_task_a_english()

    model = train_transformer(
        model_name=MODEL_MAP["arabic"],
        train_df=en_train,
        dev_df=en_dev,
        num_labels=2,
        language="english",
        mode="finetune",
        few_shot_k=None,
        config=config,
        device=device,
        return_model=True
    )

    print("Stage 2: Arabic few-shot fine-tuning...")
    ar_train, ar_dev = load_task_a_arabic()

    train_transformer(
        model_name=MODEL_MAP["arabic"],
        train_df=ar_train,
        dev_df=ar_dev,
        num_labels=2,
        language="arabic",
        mode="few-shot",
        few_shot_k=args.k,
        config=config,
        device=device,
        model=model
    )


if __name__ == "__main__":
    main()