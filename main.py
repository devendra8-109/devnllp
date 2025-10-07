# ==============================================================
# 🎬 Netflix-Style Advanced NLP Analyzer with SMOTE Balancing
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import time

st.set_page_config(page_title="Netflix Style NLP Analyzer", layout="wide")

# ==============================================================
# 🎭 Sidebar
# ==============================================================
st.sidebar.title("🎛 Netflix NLP Control Panel")
st.sidebar.markdown("### Choose your analysis flow")
vectorizer_choice = st.sidebar.selectbox("Select Vectorization Method", ["TF-IDF", "Count Vectorizer"])
apply_smote = st.sidebar.checkbox("🔁 Apply SMOTE Balancing", value=True)
st.sidebar.markdown("---")
st.sidebar.info("Upload your dataset or use a sample.")

# ==============================================================
# 📂 Data Upload / Sample
# ==============================================================
uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        "text": [
            "I love this product, it’s amazing!",
            "This is the worst thing I ever bought.",
            "Absolutely fantastic experience.",
            "Terrible quality, do not recommend.",
            "It was okay, not great but not bad either.",
            "This is a positive review.",
            "Another negative feedback here.",
            "This is a neutral statement.",
            "Loved it!",
            "Hated it.",
            "It's alright.",
            "This is a truly positive experience.",
            "The product was extremely bad.",
            "It's neither good nor bad, just there."
        ],
        "label": [
            "positive", "negative", "positive", "negative", "neutral",
            "positive", "negative", "neutral", "positive", "negative", "neutral",
            "positive", "negative", "neutral"
        ]
    })
    st.info("Using sample dataset.")

st.dataframe(df.head())

# ==============================================================
# 🧩 NLP Feature Extraction
# ==============================================================
st.subheader("🔍 Step 1: NLP Feature Extraction")
nlp = spacy.load("en_core_web_sm")

def extract_features(text):
    doc = nlp(text)
    polarity = TextBlob(text).sentiment.polarity
    subjectivity = TextBlob(text).sentiment.subjectivity
    avg_token_length = np.mean([len(t.text) for t in doc])
    pos_ratio = len([t for t in doc if t.pos_ == "ADJ"]) / (len(doc) + 1)
    return polarity, subjectivity, avg_token_length, pos_ratio

with st.spinner("Extracting linguistic features..."):
    features = df["text"].apply(lambda x: extract_features(str(x)))
    df_features = pd.DataFrame(features.tolist(),
                               columns=["polarity", "subjectivity", "avg_token_length", "pos_ratio"])
    df = pd.concat([df, df_features], axis=1)

st.success("Features extracted successfully ✅")
st.dataframe(df.head())

# ==============================================================
# 🧠 Vectorization
# ==============================================================
st.subheader("🧮 Step 2: Text Vectorization")

if vectorizer_choice == "TF-IDF":
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
else:
    vectorizer = CountVectorizer(stop_words="english", max_features=500)

X_text = vectorizer.fit_transform(df["text"])
X_nlp = df[["polarity", "subjectivity", "avg_token_length", "pos_ratio"]].values
X_combined = np.hstack((X_text.toarray(), X_nlp))
y = df["label"]

st.write(f"✅ Vectorization complete using **{vectorizer_choice}**")

# ==============================================================
# 🧩 Netflix Style Model Trainer
# ==============================================================

class NetflixModelTrainer:
    def __init__(self):
        self.models = {
            # "Naive Bayes": MultinomialNB(), # Removed due to negative values in combined features
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
            "SVM": SVC(kernel='linear', probability=True, random_state=42)
        }

    def train_and_evaluate(self, X, y, use_smote=True):
        """Train models with optional SMOTE balancing"""
        results = {}

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)

        test_size = max(0.15, min(0.25, 3 * n_classes / len(y_encoded)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # ✅ Apply SMOTE
        if use_smote:
            try:
                # Adjust k_neighbors based on the number of samples in the smallest class after train/test split
                min_samples_in_smallest_class = min(np.bincount(y_train))
                smote_k_neighbors = min(5, min_samples_in_smallest_class - 1) if min_samples_in_smallest_class > 1 else 0

                if smote_k_neighbors > 0:
                    smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    st.info(f"✅ Applied SMOTE: Training data balanced to {len(y_train)} samples.")
                else:
                    st.warning("⚠️ SMOTE skipped: Not enough samples in the smallest class to apply SMOTE.")

            except Exception as e:
                st.warning(f"⚠️ SMOTE skipped due to: {str(e)}")

        progress_container = st.empty()

        for i, (name, model) in enumerate(self.models.items()):
            with progress_container.container():
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**🎬 Training {name}...**")
                with cols[1]:
                    progress_bar = st.progress(0)
                    for step in range(5):
                        progress_bar.progress((step + 1) / 5)
                        time.sleep(0.1)

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

                results[name] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    "conf_matrix": confusion_matrix(y_test, y_pred),
                    "report": classification_report(y_test, y_pred, output_dict=True),
                    "model": model,
                    "true": y_test,
                    "pred": y_pred
                }
            except Exception as e:
                results[name] = {"error": str(e)}

        progress_container.empty()
        return results, le

# ==============================================================
# 🎬 Model Training
# ==============================================================
st.subheader("🎯 Step 3: Train & Evaluate Models")
trainer = NetflixModelTrainer()
results, le = trainer.train_and_evaluate(X_combined, y, use_smote=apply_smote)

# ==============================================================
# 📊 Results Visualization
# ==============================================================
st.subheader("📈 Step 4: Results Dashboard")

scores_df = pd.DataFrame([
    {"Model": k,
     "Accuracy": v["accuracy"],
     "Precision": v["precision"],
     "Recall": v["recall"],
     "F1 Score": v["f1_score"]}
    for k, v in results.items() if "accuracy" in v
])

st.dataframe(scores_df.style.highlight_max(color="lightgreen", axis=0))

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=scores_df.melt(id_vars="Model"), x="Model", y="value", hue="variable", ax=ax)
plt.title("📊 Model Performance Comparison")
plt.ylabel("Score")
st.pyplot(fig)

# ==============================================================
# 🧩 Confusion Matrix Viewer
# ==============================================================
st.subheader("🔍 Step 5: Confusion Matrix Explorer")
selected_model = st.selectbox("Select Model to View Confusion Matrix", scores_df["Model"])
if selected_model in results:
    cm = results[selected_model]["conf_matrix"]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix - {selected_model}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

st.success("✅ NLP Model Training & Evaluation Completed Successfully!")

# ==============================================================
# 🧠 End of Script
# ==============================================================
