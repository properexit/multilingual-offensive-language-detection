from models.baseline import BaselineModel
from datasets.english.loaders import load_task_a_english


def main():
    train_df, dev_df = load_task_a_english()

    model = BaselineModel()
    model.fit(train_df["tweet"], train_df["label"])
    model.evaluate(
        dev_df["tweet"],
        dev_df["label"],
        target_names=["NOT", "OFF"]
    )


if __name__ == "__main__":
    main()