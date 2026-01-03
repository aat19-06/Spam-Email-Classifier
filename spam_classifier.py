import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# 1. Example dataset: SMS Spam Collection (download from UCI or Kaggle)
# Format: label (ham/spam), message
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]

# Convert labels to binary
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 3. Text vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Model
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test_vec)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Example inference
sample = ["Congratulations! You won a free iPhone. Click here."]
sample_vec = vectorizer.transform(sample)
pred = clf.predict(sample_vec)[0]
print("Prediction:", "Spam" if pred == 1 else "Ham")