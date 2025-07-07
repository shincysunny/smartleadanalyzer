import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("gpt_labeled_leads.csv")

# Manual features
df["desc_length"] = df["Description"].apply(lambda x: len(str(x).split()))
df["has_keywords"] = df["Description"].str.contains(
    r"\b(?:AI|ML|SaaS|platform|automation|startup|insight)\b", case=False
).astype(int)
df["is_generic"] = df["Description"].str.contains(
    r"\b(?:simple|note|list|tracker|reminder|habit|task|basic|chat)\b", case=False
).astype(int)

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Lead Tag"])
joblib.dump(label_encoder, "encoder_bert.joblib")

# Sentence-BERT embeddings
print("🔍 Encoding descriptions with Sentence-BERT...")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
X_text = bert_model.encode(df["Description"].tolist(), show_progress_bar=True)

# Manual features (scaled)
manual_features = df[["desc_length", "has_keywords", "is_generic"]].values
scaler = StandardScaler()
manual_scaled = scaler.fit_transform(manual_features)
joblib.dump(scaler, "scaler_bert.joblib")

# Combine features
X = np.hstack([X_text, manual_scaled])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
clf = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"\n✅ Model Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(clf, "lead_classifier_bert.joblib")
print("\n💾 Model saved as 'lead_classifier_bert.joblib'")
