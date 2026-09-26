import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("data/divorce_data_large.csv")
X, y = data["text"], data["outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)
model = Pipeline([("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
                  ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred))
