from sklearn.metrics import f1_score, classification_report


def evaluate_classification(y_true, y_pred, target_names=None):
    """
    Prints Macro F1 and full classification report.
    """
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0
        )
    )