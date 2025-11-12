####################################################################
# netflix_nlp_full.py
# Netflix-Style Advanced NLP Analyzer — SMOTE + Google Drive support
# Run: streamlit run netflix_nlp_full.py
# In Colab: install dependencies (commented installs), mount drive, then run streamlit.
####################################################################

# -----------------------
# (Optional) Installs (uncomment in Colab)
# -----------------------
!pip install streamlit scikit-learn imbalanced-learn textblob spacy seaborn matplotlib pyngrok
!python -m spacy download en_core_web_sm

import os
import sys
import io
import json
import tempfile
import pickle
import platform
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# sklearn / imblearn / nlp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# optional installs may provide these; wrap in try for nicer error messages
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
except Exception:
    spacy = None

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

# Google Colab/Drive support (safe import)
try:
    from google.colab import drive as colab_drive  # only available in Colab
    IN_COLAB = True
except Exception:
    colab_drive = None
    IN_COLAB = False

# pyngrok for exposing Streamlit in Colab (optional)
try:
    from pyngrok import ngrok
except Exception:
    ngrok = None

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Netflix Style NLP Analyzer", layout="wide")
st.title("🎬 Netflix-Style Advanced NLP Analyzer — Full (SMOTE + Google Drive)")

# -----------------------
# Sidebar: options
# -----------------------
st.sidebar.header("🔧 Options")
run_mode = st.sidebar.selectbox("Run mode", ["Local / Upload", "Google Drive (Colab)"])
vectorizer_choice = st.sidebar.selectbox("Vectorizer", ["TF-IDF", "Count Vectorizer"])
apply_smote = st.sidebar.checkbox("Apply SMOTE (class balancing)", value=True)
max_features = st.sidebar.number_input("Max features (vectorizer)", value=1000, min_value=100, max_value=20000, step=100)
sample_preview = st.sidebar.checkbox("Show dataset preview", value=True)
save_artifacts = st.sidebar.checkbox("Save trained models & vectorizer", value=True)

# -----------------------
# Helper: show install hints if libs missing
# -----------------------
def check_optional_libs():
    missing = []
    if SMOTE is None:
        missing.append("imbalanced-learn (SMOTE)")
    if spacy is None or TextBlob is None:
        missing.append("spaCy or TextBlob")
    if missing:
        st.warning("Optional libraries missing: " + ", ".join(missing) +
                   ". If you want NLP features or SMOTE, install them (see top of script).")

check_optional_libs()

# -----------------------
# Data loading: either upload or Google Drive path
# -----------------------
df = None
uploaded_file = None
if run_mode == "Local / Upload":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV loaded from upload.")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
else:
    st.info("Google Drive mode selected. (Mount Drive in Colab before running.)")
    drive_path_input = st.text_input("Enter full path to CSV on Google Drive or local FS (e.g. /content/drive/MyDrive/data.csv or C:\\Users\\You\\Downloads\\file.csv)", value="")
    if st.button("Load path"):
        if drive_path_input:
            try:
                df = pd.read_csv(drive_path_input)
                st.success(f"CSV loaded from path: {drive_path_input}")
            except Exception as e:
                st.error(f"Error loading CSV from path: {e}")

# If no file yet, provide sample dataset
if df is None:
    st.info("No dataset loaded — using built-in sample dataset.")
    df = pd.DataFrame({
        "text": [
            "I love this product, it’s amazing!",
            "This is the worst thing I ever bought.",
            "Absolutely fantastic experience.",
            "Terrible quality, do not recommend.",
            "It was okay, not great but not bad either."
        ],
        "label": ["positive", "negative", "positive", "negative", "neutral"]
    })

if sample_preview:
    st.subheader("Dataset preview")
    st.dataframe(df.head())

# -----------------------
# Auto-detect text and label columns (or let user pick)
# -----------------------
st.subheader("Column selection")

possible_text_cols = ['text', 'statement', 'content', 'message', 'review', 'tweet', 'comment']
possible_label_cols = ['label', 'sentiment', 'class', 'category', 'target', 'verdict', 'truthfulness']

text_col = next((c for c in df.columns if c.lower() in possible_text_cols), None)
label_col = next((c for c in df.columns if c.lower() in possible_label_cols), None)

# Show detected and allow override
st.write("Auto-detected text column:", text_col)
st.write("Auto-detected label column:", label_col)

text_col = st.selectbox("Choose text column", options=list(df.columns), index=(list(df.columns).index(text_col) if text_col in df.columns else 0))
label_col = st.selectbox("Choose label column", options=list(df.columns), index=(list(df.columns).index(label_col) if label_col in df.columns else 1 if len(df.columns)>1 else 0))

# Normalize names
df = df.rename(columns={text_col: "text", label_col: "label"})
df = df.dropna(subset=["text", "label"])
st.success(f"Using 'text' <- {text_col}, 'label' <- {label_col}")

# -----------------------
# If in Colab and user wants to mount drive, show button
# -----------------------
if IN_COLAB and run_mode == "Google Drive (Colab) and not df is None":
    # Note: This is informational — mounting must be done by running drive.mount separately in a Colab cell.
    st.info("You are in Colab environment. If your file is on Drive, run `drive.mount('/content/drive')` in a notebook cell then provide the path above.")

# -----------------------
# NLP Feature extraction (spaCy + TextBlob) — optional if libs installed
# -----------------------
st.subheader("NLP features (optional)")
use_nlp_features = st.checkbox("Extract NLP features (polarity, subjectivity, avg token length, adj ratio)", value=True)

if use_nlp_features:
    if spacy is None or TextBlob is None:
        st.error("spaCy or TextBlob not installed. Turn off NLP features or install 'spacy' and 'textblob'.")
        use_nlp_features = False
    else:
        # load spaCy model (may be heavy; load once)
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            st.warning("spaCy model 'en_core_web_sm' not found. Attempt to download or load manually in your environment.")
            raise

        @st.cache_data(show_spinner=False)
        def compute_nlp_features(series):
            records = []
            for t in series.astype(str).tolist():
                doc = nlp(t)
                polarity = TextBlob(t).sentiment.polarity
                subjectivity = TextBlob(t).sentiment.subjectivity
                avg_token_length = np.mean([len(tok.text) for tok in doc]) if len(doc)>0 else 0.0
                pos_ratio = len([tok for tok in doc if tok.pos_ == "ADJ"]) / (len(doc) + 1)
                records.append((polarity, subjectivity, avg_token_length, pos_ratio))
            return pd.DataFrame(records, columns=["polarity","subjectivity","avg_token_length","pos_ratio"])

        with st.spinner("Extracting NLP features..."):
            nlp_df = compute_nlp_features(df["text"])
            df = pd.concat([df.reset_index(drop=True), nlp_df.reset_index(drop=True)], axis=1)
        st.success("NLP features added.")
        if sample_preview:
            st.dataframe(df.head())

# -----------------------
# Vectorization
# -----------------------
st.subheader("Vectorization")
vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features) if vectorizer_choice == "TF-IDF" else CountVectorizer(stop_words="english", max_features=max_features)
with st.spinner(f"Fitting {vectorizer_choice} vectorizer..."):
    X_text = vectorizer.fit_transform(df["text"].astype(str))
st.success("Vectorization completed.")

# Combine text vectors + optional numeric NLP features
if use_nlp_features:
    X_nlp = df[["polarity","subjectivity","avg_token_length","pos_ratio"]].values
    X_combined = np.hstack([X_text.toarray(), X_nlp])
else:
    X_combined = X_text.toarray()

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(df["label"].astype(str))

st.write("Label classes:", list(le.classes_))
st.write("Class distribution (before SMOTE):")
st.bar_chart(pd.Series(y_encoded).value_counts())

# -----------------------
# SMOTE balancing
# -----------------------
if apply_smote and SMOTE is None:
    st.warning("SMOTE requested but imbalanced-learn not installed. Proceeding without SMOTE.")
    apply_smote = False

if apply_smote:
    try:
        with st.spinner("Applying SMOTE to balance classes..."):
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_combined, y_encoded)
        st.success("SMOTE applied successfully.")
        st.write("Class distribution (after SMOTE):")
        st.bar_chart(pd.Series(y_res).value_counts())
    except Exception as e:
        st.warning(f"SMOTE failed: {e}. Proceeding without SMOTE.")
        X_res, y_res = X_combined, y_encoded
else:
    X_res, y_res = X_combined, y_encoded

# -----------------------
# Model training
# -----------------------
st.subheader("Model training & evaluation")

# Choose models (you can extend)
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42)
}

test_size = st.sidebar.slider("Test size (fraction)", 0.1, 0.4, 0.25, 0.05)
do_train = st.button("Train models")

results = {}
if do_train:
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_size, random_state=42, stratify=y_res)
    prog = st.progress(0)
    for i, (name, model) in enumerate(models.items()):
        st.markdown(f"### Training: {name}")
        # small progress simulation
        for p in range(5):
            prog.progress(min(((i*5+p+1)/(len(models)*5)), 1.0))
            time.sleep(0.05)
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                "model": model,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                "conf_matrix": confusion_matrix(y_test, y_pred),
                "report": classification_report(y_test, y_pred, output_dict=True),
                "y_test": y_test,
                "y_pred": y_pred
            }
            st.success(f"{name} trained. Accuracy: {results[name]['accuracy']:.3f}")
        except Exception as e:
            st.error(f"Error training {name}: {e}")
    prog.empty()

# -----------------------
# Show results table & plots
# -----------------------
if results:
    scores_df = pd.DataFrame([{
        "Model": name,
        "Accuracy": stats["accuracy"],
        "Precision": stats["precision"],
        "Recall": stats["recall"],
        "F1 Score": stats["f1_score"]
    } for name, stats in results.items()]).sort_values("F1 Score", ascending=False).reset_index(drop=True)

    st.subheader("Model comparison")
    st.dataframe(scores_df.style.highlight_max(axis=0))

    # barplot
    fig, ax = plt.subplots(figsize=(9,5))
    sns.barplot(data=scores_df.melt(id_vars="Model"), x="Model", y="value", hue="variable", ax=ax)
    ax.set_title("Model performance comparison")
    st.pyplot(fig)

    # confusion matrix selector
    st.subheader("Confusion Matrix")
    chosen = st.selectbox("Choose model", options=list(results.keys()))
    cm = results[chosen]["conf_matrix"]
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax2,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title(f"Confusion matrix - {chosen}")
    st.pyplot(fig2)

    # show full classification report for top model
    st.subheader("Detailed classification report (top model)")
    top_model_name = scores_df.iloc[0]["Model"]
    st.write(f"Top model: {top_model_name}")
    st.text(classification_report(results[top_model_name]["y_test"], results[top_model_name]["y_pred"], target_names=le.classes_))

    # -----------------------
    # Save artifacts: vectorizer & top model
    # -----------------------
    if save_artifacts:
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # save vectorizer
        vec_path = artifacts_dir / "vectorizer.pkl"
        with open(vec_path, "wb") as f:
            pickle.dump(vectorizer, f)

        # save label encoder
        le_path = artifacts_dir / "label_encoder.pkl"
        with open(le_path, "wb") as f:
            pickle.dump(le, f)

        # save each trained model
        for name, info in results.items():
            model_file = artifacts_dir / f"model_{name.replace(' ','_')}.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(info["model"], f)

        st.success(f"Saved artifacts to {artifacts_dir.resolve().as_posix()}")
        st.markdown("Download artifacts:")
        with open(vec_path, "rb") as f:
            st.download_button("Download vectorizer.pkl", data=f, file_name="vectorizer.pkl")
        with open(le_path, "rb") as f:
            st.download_button("Download label_encoder.pkl", data=f, file_name="label_encoder.pkl")
        for name in results:
            model_file = artifacts_dir / f"model_{name.replace(' ','_')}.pkl"
            with open(model_file, "rb") as f:
                st.download_button(f"Download {name} model", data=f, file_name=f"{name.replace(' ','_')}.pkl")

# -----------------------
# Single text prediction using top model (if available)
# -----------------------
st.subheader("Single text prediction (use trained top model)")
input_text = st.text_area("Enter text to classify")
predict_button = st.button("Predict")

if predict_button:
    if not results:
        st.error("No trained model available. Train models first.")
    else:
        top_name = scores_df.iloc[0]["Model"]
        top_model = results[top_name]["model"]
        # vectorize input — combine features if used
        x_text = vectorizer.transform([input_text]).toarray()
        if use_nlp_features:
            # compute features quickly with spaCy/TextBlob (small)
            doc = nlp(input_text)
            polarity = TextBlob(input_text).sentiment.polarity
            subjectivity = TextBlob(input_text).sentiment.subjectivity
            avg_token_length = np.mean([len(tok.text) for tok in doc]) if len(doc)>0 else 0
            pos_ratio = len([tok for tok in doc if tok.pos_ == "ADJ"]) / (len(doc)+1)
            x_comb = np.hstack([x_text, np.array([[polarity, subjectivity, avg_token_length, pos_ratio]])])
        else:
            x_comb = x_text
        pred_code = top_model.predict(x_comb)[0]
        pred_label = le.inverse_transform([pred_code])[0]
        st.success(f"Predicted label: {pred_label} (model: {top_name})")

# -----------------------
# Optional: show raw dataset / allow download of processed dataset
# -----------------------
st.subheader("Processed dataset")
if st.checkbox("Show processed dataframe"):
    st.dataframe(df.head(200))

if st.button("Download processed dataset as CSV"):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="processed_dataset.csv")

# -----------------------
# Notes: Colab usage
# -----------------------
if IN_COLAB:
    st.info("You're in Colab. If you want to mount Google Drive, run in a separate Colab cell:\n\nfrom google.colab import drive\ndrive.mount('/content/drive')\n\nThen use the 'Google Drive (Colab)' run mode and provide the CSV path (eg /content/drive/MyDrive/your.csv).")

# -----------------------
# End
# -----------------------
st.info("Script end. For heavy datasets, consider raising max_features or disabling NLP features to speed up runs.")
