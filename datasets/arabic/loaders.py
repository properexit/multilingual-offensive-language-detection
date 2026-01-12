import pandas as pd
from sklearn.model_selection import train_test_split


def load_task_a_arabic(split_ratio=0.2):
    """
    Arabic OffensEval 2020 provides labels ONLY for Task A.
    TSV format:
    id <TAB> tweet <TAB> subtask_a
    """

    tweets, labels = [], []

    with open(
        "data/raw/arabic/offenseval-ar-training-v1/offenseval-ar-training-v1.tsv",
        encoding="utf-8"
    ) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            _, tweet, sub_a = parts
            tweets.append(tweet.replace("<LF>", " "))
            labels.append(1 if sub_a == "OFF" else 0)

    df = pd.DataFrame({
        "tweet": tweets,
        "label": labels
    })

    return train_test_split(
        df,
        test_size=split_ratio,
        stratify=df["label"],
        random_state=42
    )