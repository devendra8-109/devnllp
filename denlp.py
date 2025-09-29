# ============================================
# 📌 Streamlit NLP Phase-wise with All Models (Enhanced Visuals)
# ============================================

import streamlit as st
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ============================
# Load SpaCy & Globals
# ============================
nlp = spacy.load("en_core_web_sm")
stop_words = STOP_WORDS

# ============================
# Phase Feature Extractors
# ============================
def lexical_preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

def syntactic_features(text):
    doc = nlp(text)
    return " ".join([token.pos_ for token in doc])

def semantic_features(text):
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split()) > 0])}"

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]
def pragmatic_features(text):
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# Train & Evaluate All Models
# ============================
def evaluate_models(X_features, y):
    results = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC()
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            results[name] = round(acc, 2)
        except Exception as e:
            results[name] = f"Error: {str(e)}"

    return results

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="NLP Phase Analysis", layout="wide")

st.title("📊 Rumor Buster - Enhanced Edition")

uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    df = pd.read_csv(uploaded_file)

    st.markdown("### ⚙️ Configuration")
    text_col = st.selectbox("Select Text Column:", df.columns)
    target_col = st.selectbox("Select Target Column:", df.columns)

    phase = st.selectbox("Select NLP Phase:", [
        "Lexical & Morphological",
        "Syntactic",
        "Semantic",
        "Discourse",
        "Pragmatic"
    ])

    run_analysis = st.button("🚀 Run Analysis", type="primary")

    st.write("### 📋 Data Preview")
    with st.expander("👀 Click to view sample data"):
        st.dataframe(df.head(), use_container_width=True)

    if run_analysis:
        st.write(f"### 🔍 Investigation: {phase}")
        
        with st.spinner("Analyzing... please wait ⏳"):
            X = df[text_col].astype(str)
            y = df[target_col]

            if phase == "Lexical & Morphological":
                X_processed = X.apply(lexical_preprocess)
                X_features = CountVectorizer().fit_transform(X_processed)
                # WordCloud
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(X_processed))
                fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                ax_wc.imshow(wordcloud, interpolation="bilinear")
                ax_wc.axis("off")

            elif phase == "Syntactic":
                X_processed = X.apply(syntactic_features)
                X_features = CountVectorizer().fit_transform(X_processed)

            elif phase == "Semantic":
                X_features = pd.DataFrame(X.apply(semantic_features).tolist(),
                                          columns=["polarity", "subjectivity"])

            elif phase == "Discourse":
                X_processed = X.apply(discourse_features)
                X_features = CountVectorizer().fit_transform(X_processed)

            elif phase == "Pragmatic":
                X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(),
                                          columns=pragmatic_words)

            results = evaluate_models(X_features, y)

        results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
        best_idx = results_df["Accuracy"].idxmax()

        # Tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📊 Charts", "📈 Metrics", "📋 Detailed Results"])

        with tab1:
            st.subheader("Model Performance")
            fig_bar = px.bar(
                results_df,
                x="Model",
                y="Accuracy",
                color="Model",
                text="Accuracy",
                title=f"Model Performance - {phase}",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_pie = px.pie(
                results_df,
                values="Accuracy",
                names="Model",
                title="Performance Distribution",
                hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            if phase == "Lexical & Morphological":
                st.subheader("☁️ WordCloud")
                st.pyplot(fig_wc)

        with tab2:
            st.subheader("🏆 Operational Benchmarks")
            cols = st.columns(4)
            for idx, (model, accuracy) in enumerate(zip(results_df["Model"], results_df["Accuracy"])):
                with cols[idx % 4]:
                    if idx == best_idx:
                        st.metric(f"🥇 {model}", f"{accuracy:.1f}%", "Best")
                    else:
                        delta = round(accuracy - results_df.loc[best_idx, "Accuracy"], 1)
                        st.metric(model, f"{accuracy:.1f}%", f"{delta}%" if delta != 0 else "")

        with tab3:
            st.subheader("📋 The Inside Scoop")
            results_display = results_df.copy()
            results_display["Accuracy"] = results_display["Accuracy"].apply(lambda x: f"{x:.1f}%")
            results_display["Rank"] = range(1, len(results_display) + 1)
            st.dataframe(results_display[["Rank", "Model", "Accuracy"]], use_container_width=True)

else:
    st.info("👆 Please upload a CSV to begin analysis.")
