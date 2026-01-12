import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from utils.metrics import evaluate_classification
from models.transformer import load_transformer
from datasets.english.dataset import TweetDataset
from datasets.arabic.dataset import ArabicTweetDataset


def train_transformer(
    model_name,
    train_df,
    dev_df,
    num_labels,
    language,
    mode,
    few_shot_k,
    config,
    device,
    model=None,
    return_model=False
):
    """
    Unified transformer training loop.
    Supports:
    - English finetuning
    - Arabic zero-shot
    - Arabic few-shot with English pretraining (via model reuse)
    """

    # ------------------
    # CONFIG
    # ------------------
    batch_size = int(config.get("batch_size", 8))
    epochs = int(config.get("epochs", 1))
    lr = float(config.get("learning_rate", 2e-5))
    max_len = int(config.get("max_length", 128))
    class_weighted = config.get("class_weighted", False)

    # ------------------
    # TOKENIZER + MODEL
    # ------------------
    if model is None:
        tokenizer, model = load_transformer(model_name, num_labels)
        model.to(device)
    else:
        tokenizer = load_transformer(model_name, num_labels)[0]

    optimizer = AdamW(model.parameters(), lr=lr)

    Dataset = TweetDataset if language == "english" else ArabicTweetDataset

    # ------------------
    # FEW-SHOT SAMPLING
    # ------------------
    if mode == "few-shot" and few_shot_k is not None:
        train_df, _ = train_test_split(
            train_df,
            train_size=few_shot_k,
            stratify=train_df["label"],
            random_state=42
        )

    train_ds = Dataset(train_df["tweet"], train_df["label"], tokenizer, max_len)
    dev_ds = Dataset(dev_df["tweet"], dev_df["label"], tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)

    # ------------------
    # LOSS FUNCTION
    # ------------------
    if class_weighted:
        weights = compute_class_weight(
            "balanced",
            classes=np.unique(train_df["label"]),
            y=train_df["label"]
        )
        weights = torch.tensor(weights, dtype=torch.float).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # ------------------
    # TRAINING
    # ------------------
    if mode != "zero-shot":
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()

                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)
                ).logits

                loss = loss_fn(logits, batch["labels"].to(device))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

    # ------------------
    # EVALUATION
    # ------------------
    model.eval()
    preds, gold = [], []

    with torch.no_grad():
        for batch in dev_loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            ).logits

            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            gold.extend(batch["labels"].numpy())

    evaluate_classification(gold, preds)

    if return_model:
        return model