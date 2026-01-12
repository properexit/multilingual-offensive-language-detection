from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report


class BaselineModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000
        )
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)

    def evaluate(self, texts, labels, target_names=None):
        X = self.vectorizer.transform(texts)
        preds = self.clf.predict(X)

        print("Macro F1:", f1_score(labels, preds, average="macro"))
        print(
            classification_report(
                labels,
                preds,
                target_names=target_names,
                zero_division=0
            )
        )