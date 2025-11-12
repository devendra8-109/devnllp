# ============================================
# 🎬 NLP Analysis Suite with SMOTE
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="NLP Analyzer Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# Netflix Style CSS
# ============================
st.markdown("""
<style>
    /* Netflix Color Scheme */
    :root {
        --netflix-red: #e50914;
        --netflix-dark: #141414;
        --netflix-black: #000000;
        --netflix-gray: #2f2f2f;
        --netflix-light: #f5f5f1;
        --netflix-white: #ffffff;
        --netflix-card: #181818;
    }
    
    /* Main background */
    .stApp {
        background: var(--netflix-black);
        color: var(--netflix-white);
    }
    
    /* Netflix Header */
    .netflix-header {
        background: linear-gradient(180deg, rgba(0,0,0,0.8) 0%, transparent 100%);
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .netflix-card {
        background: var(--netflix-card);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .netflix-card:hover {
        transform: scale(1.02);
        border-color: var(--netflix-red);
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.3);
    }
    
    .netflix-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--netflix-red);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, var(--netflix-red) 0%, #b20710 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
        border: none;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Sections */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--netflix-white);
        margin: 3rem 0 1.5rem 0;
        padding-left: 1rem;
        border-left: 4px solid var(--netflix-red);
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Sidebar - Netflix Style */
    .css-1d391kg, .css-1lcbmhc {
        background: var(--netflix-dark) !important;
        border-right: 1px solid var(--netflix-gray) !important;
    }
    
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--netflix-red);
        margin-bottom: 1.5rem;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Buttons - Netflix Style */
    .stButton button {
        width: 100%;
        background: var(--netflix-red);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 4px;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton button:hover {
        background: #b20710;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(229, 9, 20, 0.4);
    }
    
    /* Select boxes and inputs */
    .stSelectbox, .stTextInput, .stNumberInput {
        background: var(--netflix-card) !important;
        color: white !important;
        border: 1px solid var(--netflix-gray) !important;
    }
    
    .stSelectbox div, .stTextInput input, .stNumberInput input {
        background: var(--netflix-card) !important;
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--netflix-dark);
        border-bottom: 2px solid var(--netflix-gray);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--netflix-dark) !important;
        color: var(--netflix-light) !important;
        border-radius: 0;
        padding: 1rem 2rem;
        border-bottom: 3px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--netflix-dark) !important;
        color: var(--netflix-red) !important;
        border-bottom: 3px solid var(--netflix-red) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--netflix-card) !important;
        color: var(--netflix-white) !important;
        border: 1px solid var(--netflix-gray) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: var(--netflix-red);
    }
    
    /* Success, Error, Info */
    .stSuccess {
        background: rgba(0, 255, 0, 0.1) !important;
        border: 1px solid #00ff00 !important;
        color: #00ff00 !important;
    }
    
    .stError {
        background: rgba(229, 9, 20, 0.1) !important;
        border: 1px solid var(--netflix-red) !important;
        color: var(--netflix-red) !important;
    }
    
    .stInfo {
        background: rgba(0, 191, 255, 0.1) !important;
        border: 1px solid #00bfff !important;
        color: #00bfff !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: var(--netflix-card) !important;
        color: white !important;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(20,20,20,0.9) 0%, rgba(0,0,0,0.9) 100%), 
                    url('https://images.unsplash.com/photo-1489599809505-fb40ebc6fbc1?ixlib=rb-4.0.3') center/cover;
        padding: 4rem 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        text-align: center;
        border: 1px solid var(--netflix-gray);
    }
    
    /* Model Performance Cards */
    .model-card {
        background: var(--netflix-card);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        border-color: var(--netflix-red);
        transform: translateY(-5px);
    }
    
    .model-accuracy {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--netflix-red);
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Feature Tags */
    .feature-tag {
        background: rgba(229, 9, 20, 0.2);
        color: var(--netflix-red);
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        display: inline-block;
        border: 1px solid rgba(229, 9, 20, 0.3);
    }
    
    /* SMOTE Badge */
    .smote-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 0.5rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Initialize NLP
# ============================
@st.cache_resource
def load_nlp_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("""
        **SpaCy English model not found.** 
        Please install: `python -m spacy download en_core_web_sm`
        """)
        st.stop()

nlp = load_nlp_model()
stop_words = STOP_WORDS

# ============================
# Feature Engineering Classes
# ============================
class NetflixFeatureExtractor:
    @staticmethod
    def extract_lexical_features(texts):
        """Extract lexical features with advanced preprocessing"""
        processed_texts = []
        for text in texts:
            doc = nlp(str(text).lower())
            tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
            processed_texts.append(" ".join(tokens))
        return TfidfVectorizer(max_features=1000, ngram_range=(1, 2)).fit_transform(processed_texts)
    
    @staticmethod
    def extract_semantic_features(texts):
        """Extract semantic features with sentiment analysis"""
        features = []
        for text in texts:
            blob = TextBlob(str(text))
            features.append([
                blob.sentiment.polarity,
                blob.sentiment.subjectivity,
                len(text.split()),
                len([word for word in text.split() if len(word) > 6]),
            ])
        return np.array(features)
    
    @staticmethod
    def extract_syntactic_features(texts):
        """Extract syntactic features with POS analysis"""
        processed_texts = []
        for text in texts:
            doc = nlp(str(text))
            pos_tags = [f"{token.pos_}_{token.tag_}" for token in doc]
            processed_texts.append(" ".join(pos_tags))
        return CountVectorizer(max_features=800, ngram_range=(1, 3)).fit_transform(processed_texts)
    
    @staticmethod
    def extract_pragmatic_features(texts):
        """Extract pragmatic features - context and intent analysis"""
        pragmatic_features = []
        pragmatic_indicators = {
            'modality': ['must', 'should', 'could', 'would', 'might', 'may'],
            'certainty': ['certainly', 'definitely', 'obviously', 'clearly'],
            'uncertainty': ['perhaps', 'maybe', 'possibly', 'probably'],
            'question': ['what', 'why', 'how', 'when', 'where', 'which', '?'],
            'emphasis': ['very', 'extremely', 'highly', 'absolutely']
        }
        
        for text in texts:
            text_lower = str(text).lower()
            features = []
            
            for category, words in pragmatic_indicators.items():
                count = sum(text_lower.count(word) for word in words)
                features.append(count)
            
            features.extend([
                text.count('!'),
                text.count('?'),
                len([s for s in text.split('.') if s.strip()]),
                len([w for w in text.split() if w.istitle()]),
            ])
            
            pragmatic_features.append(features)
        
        return np.array(pragmatic_features)

# ============================
# Netflix Style Model Trainer with SMOTE
# ============================
class NetflixModelTrainer:
    def __init__(self, use_smote=True):
        self.use_smote = use_smote
        self.models = {
            "🎬 Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "🌲 Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced'),
            "⚡ Support Vector": SVC(random_state=42, probability=True, class_weight='balanced'),
            "📊 Naive Bayes": MultinomialNB()
        }
    
    def analyze_class_distribution(self, y):
        """Analyze and visualize class distribution"""
        class_counts = pd.Series(y).value_counts()
        
        # Create class distribution visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#000000')
        
        # Bar chart
        colors = ['#e50914', '#b20710', '#8c0610', '#660208', '#400104']
        bars = ax1.bar(range(len(class_counts)), class_counts.values, color=colors[:len(class_counts)])
        ax1.set_facecolor('#141414')
        ax1.set_title('📊 Class Distribution', fontweight='bold', color='white', fontsize=14)
        ax1.set_xlabel('Classes', fontweight='bold', color='white')
        ax1.set_ylabel('Count', fontweight='bold', color='white')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax1.set_xticks(range(len(class_counts)))
        ax1.set_xticklabels([f'Class {i}' for i in range(len(class_counts))], rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', color='white')
        
        # Pie chart
        if len(class_counts) > 1:
            colors_pie = plt.cm.Reds(np.linspace(0.4, 0.8, len(class_counts)))
            wedges, texts, autotexts = ax2.pie(class_counts.values, labels=class_counts.index, 
                                             autopct='%1.1f%%', colors=colors_pie, startangle=90)
            ax2.set_title('🎯 Class Proportions', fontweight='bold', color='white', fontsize=14)
            
            for text in texts:
                text.set_color('white')
                text.set_fontweight('bold')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        return fig, class_counts
    
    def train_and_evaluate(self, X, y):
        """Netflix style model training with SMOTE and comprehensive evaluation"""
        results = {}
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)
        
        # Analyze class distribution
        distribution_fig, class_counts = self.analyze_class_distribution(y)
        
        test_size = max(0.15, min(0.25, 3 * n_classes / len(y_encoded)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Display class distribution analysis
        st.markdown("#### 📈 CLASS DISTRIBUTION ANALYSIS")
        st.pyplot(distribution_fig)
        
        # Show SMOTE recommendation
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
        
        smote_info = st.container()
        with smote_info:
            if imbalance_ratio > 2 and self.use_smote:
                st.success(f"🔧 **SMOTE Applied**: Class imbalance detected (ratio: {imbalance_ratio:.1f}:1). Generating synthetic samples...")
            elif self.use_smote:
                st.info("⚖️ **Balanced Data**: Class distribution is relatively balanced. SMOTE may provide minor improvements.")
            else:
                st.warning("⚠️ **SMOTE Disabled**: Training without synthetic data generation.")
        
        # Netflix style progress
        progress_container = st.empty()
        
        for i, (name, model) in enumerate(self.models.items()):
            with progress_container.container():
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**Training {name}**")
                    if self.use_smote:
                        st.markdown(f"<span class='smote-badge'>SMOTE ENABLED</span>", unsafe_allow_html=True)
                with cols[1]:
                    progress_bar = st.progress(0)
                    
                    # Simulate Netflix-style loading
                    for step in range(5):
                        progress_bar.progress((step + 1) / 5)
                        import time
                        time.sleep(0.1)
            
            try:
                # Apply SMOTE if enabled and there's class imbalance
                if self.use_smote and imbalance_ratio > 1.5 and n_classes > 1:
                    smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count-1))
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    
                    # Show SMOTE effect
                    if i == 0:  # Only show for first model to avoid repetition
                        st.info(f"📊 **SMOTE Effect**: Increased training samples from {len(X_train)} to {len(X_train_resampled)}")
                    
                    X_train_final, y_train_final = X_train_resampled, y_train_resampled
                else:
                    X_train_final, y_train_final = X_train, y_train
                
                model.fit(X_train_final, y_train_final)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'model': model,
                    'predictions': y_pred,
                    'true_labels': y_test,
                    'probabilities': y_proba,
                    'n_classes': n_classes,
                    'test_size': len(y_test),
                    'smote_applied': self.use_smote and imbalance_ratio > 1.5 and n_classes > 1,
                    'imbalance_ratio': imbalance_ratio
                }
                
            except Exception as e:
                results[name] = {'error': str(e)}
        
        progress_container.empty()
        return results, le

# ============================
# Netflix Style Visualizations
# ============================
class NetflixVisualizer:
    @staticmethod
    def create_performance_dashboard(results):
        """Create Netflix-style performance dashboard"""
        # Set dark theme for matplotlib
        plt.style.use('dark_background')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#000000')
        
        models = []
        metrics_data = {
            'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []
        }
        smote_status = []
        
        for model_name, result in results.items():
            if 'error' not in result:
                clean_name = model_name.replace('🎬 ', '').replace('🌲 ', '').replace('⚡ ', '').replace('📊 ', '')
                models.append(clean_name)
                metrics_data['Accuracy'].append(result['accuracy'])
                metrics_data['Precision'].append(result['precision'])
                metrics_data['Recall'].append(result['recall'])
                metrics_data['F1-Score'].append(result['f1_score'])
                smote_status.append(result.get('smote_applied', False))
        
        colors = ['#e50914' if smote else '#b20710' for smote in smote_status]
        
        # Accuracy
        bars1 = ax1.bar(models, metrics_data['Accuracy'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax1.set_facecolor('#141414')
        ax1.set_title('🎯 Accuracy' + (' (with SMOTE)' if any(smote_status) else ''), 
                     fontweight='bold', color='white', fontsize=14, pad=20)
        ax1.set_ylabel('Score', fontweight='bold', color='white')
        ax1.tick_params(axis='x', rotation=45, colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax1.grid(True, alpha=0.2, axis='y', color='white')
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')
        
        # Precision
        bars2 = ax2.bar(models, metrics_data['Precision'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax2.set_facecolor('#141414')
        ax2.set_title('📊 Precision', fontweight='bold', color='white', fontsize=14, pad=20)
        ax2.set_ylabel('Score', fontweight='bold', color='white')
        ax2.tick_params(axis='x', rotation=45, colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.grid(True, alpha=0.2, axis='y', color='white')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')
        
        # Recall
        bars3 = ax3.bar(models, metrics_data['Recall'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax3.set_facecolor('#141414')
        ax3.set_title('🔍 Recall', fontweight='bold', color='white', fontsize=14, pad=20)
        ax3.set_ylabel('Score', fontweight='bold', color='white')
        ax3.tick_params(axis='x', rotation=45, colors='white')
        ax3.tick_params(axis='y', colors='white')
        ax3.grid(True, alpha=0.2, axis='y', color='white')
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')
        
        # F1-Score
        bars4 = ax4.bar(models, metrics_data['F1-Score'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax4.set_facecolor('#141414')
        ax4.set_title('⚡ F1-Score', fontweight='bold', color='white', fontsize=14, pad=20)
        ax4.set_ylabel('Score', fontweight='bold', color='white')
        ax4.tick_params(axis='x', rotation=45, colors='white')
        ax4.tick_params(axis='y', colors='white')
        ax4.grid(True, alpha=0.2, axis='y', color='white')
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')
        
        # Add SMOTE legend
        if any(smote_status):
            fig.legend([plt.Rectangle((0,0),1,1, color='#e50914'), 
                       plt.Rectangle((0,0),1,1, color='#b20710')], 
                      ['With SMOTE', 'Without SMOTE'], 
                      loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2,
                      facecolor='#141414', edgecolor='white', fontsize=12)
        
        plt.tight_layout()
        return fig

# ============================
# Sidebar Configuration
# ============================
def setup_sidebar():
    """Setup Netflix-style sidebar"""
    st.sidebar.markdown("<div class='sidebar-header'>🔍 NLP ANALYZER PRO</div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("<div class='sidebar-header'>📁 UPLOAD DATA</div>", unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV File",
        type=["csv"],
        help="Upload your dataset for analysis"
    )
    
    # SMOTE Configuration
    st.sidebar.markdown("<div class='sidebar-header'>⚙️ ADVANCED SETTINGS</div>", unsafe_allow_html=True)
    
    use_smote = st.sidebar.checkbox(
        "Enable SMOTE", 
        value=True,
        help="Use SMOTE to handle class imbalance (recommended for better accuracy)"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_uploaded = True
            st.session_state.use_smote = use_smote
            
            st.sidebar.success(f"✅ Loaded: {df.shape[0]} rows")
            
            st.sidebar.markdown("<div class='sidebar-header'>⚙️ ANALYSIS SETUP</div>", unsafe_allow_html=True)
            
            text_col = st.sidebar.selectbox(
                "Text Column",
                df.columns,
                help="Select text data column"
            )
            
            target_col = st.sidebar.selectbox(
                "Target Column",
                df.columns,
                help="Select labels column"
            )
            
            feature_type = st.sidebar.selectbox(
                "Feature Type",
                ["Lexical", "Semantic", "Syntactic", "Pragmatic"],
                help="Choose analysis type"
            )
            
            st.session_state.config = {
                'text_col': text_col,
                'target_col': target_col,
                'feature_type': feature_type
            }
            
            if st.sidebar.button("🚀 START ANALYSIS", use_container_width=True):
                st.session_state.analyze_clicked = True
            else:
                st.session_state.analyze_clicked = False
                
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False

# ============================
# Main Content
# ============================
def main_content():
    """Main content with Netflix style"""
    
    # Netflix Header
    st.markdown("""
    <div class='netflix-header'>
        <div style='text-align: center;'>
            <h1 style='color: #e50914; font-size: 4rem; font-weight: 900; margin: 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.5);'>NLP ANALYZER PRO</h1>
            <p style='color: #f5f5f1; font-size: 1.3rem; margin: 0.5rem 0 0 0;'>Advanced Text Intelligence Platform</p>
            <div style='margin-top: 1rem;'>
                <span class="feature-tag">🤖 4 ML Algorithms</span>
                <span class="feature-tag">🔧 SMOTE Enabled</span>
                <span class="feature-tag">🎯 Pragmatic Analysis</span>
                <span class="feature-tag">📊 Real-time Analytics</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.get('file_uploaded', False):
        show_netflix_welcome()
        return
    
    df = st.session_state.df
    config = st.session_state.get('config', {})
    use_smote = st.session_state.get('use_smote', True)
    
    # Dataset Overview
    st.markdown("<div class='section-header'>📊 DATASET OVERVIEW</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[0]}</div>
            <div class="metric-label">TOTAL RECORDS</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-label">FEATURES</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.isnull().sum().sum()}</div>
            <div class="metric-label">MISSING VALUES</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        unique_classes = df[config.get('target_col', '')].nunique() if config.get('target_col') in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{unique_classes}</div>
            <div class="metric-label">UNIQUE CLASSES</div>
        </div>
        """, unsafe_allow_html=True)
    
    # SMOTE Status
    if use_smote:
        st.markdown("""
        <div class="netflix-card">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 2rem; margin-right: 1rem;">🔧</span>
                <div>
                    <h3 style="color: #e50914; margin: 0;">SMOTE ENABLED</h3>
                    <p style="color: #f5f5f1; margin: 0;">Synthetic Minority Over-sampling Technique active for handling class imbalance</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data Preview
    with st.expander("🎬 DATA PREVIEW", expanded=True):
        tab1, tab2 = st.tabs(["📋 First 10 Rows", "📈 Statistics"])
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        with tab2:
            st.write(df.describe(include='all'))
    
    # Analysis Results
    if st.session_state.get('analyze_clicked', False):
        perform_netflix_analysis(df, config, use_smote)

def show_netflix_welcome():
    """Netflix-style welcome screen"""
    st.markdown("""
    <div class='hero-section'>
        <h1 style='color: white; font-size: 3.5rem; font-weight: 900; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
            READY TO ANALYZE?
        </h1>
        <p style='color: #f5f5f1; font-size: 1.3rem; margin-bottom: 2rem;'>
            Upload your CSV file to unlock powerful text analysis capabilities
        </p>
        <div style='display: inline-flex; gap: 1rem; flex-wrap: wrap; justify-content: center;'>
            <span class="feature-tag">🤖 4 ML Algorithms</span>
            <span class="feature-tag">🔧 SMOTE Enabled</span>
            <span class="feature-tag">🎯 Pragmatic Analysis</span>
            <span class="feature-tag">📊 Real-time Analytics</span>
            <span class="feature-tag">⚡ Netflix Style</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>✨ HOW IT WORKS</div>", unsafe_allow_html=True)
    
    steps = [
        {"icon": "1️⃣", "title": "UPLOAD DATA", "desc": "Use the sidebar to upload your CSV file with text data"},
        {"icon": "2️⃣", "title": "CONFIGURE", "desc": "Select text columns, target variables, and enable SMOTE"},
        {"icon": "3️⃣", "title": "ANALYZE", "desc": "Watch as our algorithms process your data with class balancing"},
        {"icon": "4️⃣", "title": "INSIGHTS", "desc": "Get professional insights with Netflix-style visualizations"}
    ]
    
    cols = st.columns(4)
    for idx, step in enumerate(steps):
        with cols[idx]:
            st.markdown(f"""
            <div class="netflix-card">
                <div style="font-size: 2rem; margin-bottom: 1rem;">{step['icon']}</div>
                <h3 style="color: #e50914; margin-bottom: 1rem;">{step['title']}</h3>
                <p style="color: #f5f5f1; line-height: 1.5;">{step['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>🎯 FEATURE HIGHLIGHTS</div>", unsafe_allow_html=True)
    
    features = [
        {"icon": "📖", "title": "LEXICAL ANALYSIS", "desc": "Advanced word-level processing and lemmatization"},
        {"icon": "🎭", "title": "SEMANTIC INTELLIGENCE", "desc": "Sentiment analysis and meaning extraction"},
        {"icon": "🔧", "title": "SMOTE BALANCING", "desc": "Automatic class imbalance handling for better accuracy"},
        {"icon": "🔧", "title": "SYNTACTIC PROCESSING", "desc": "Grammar structure and POS analysis"},
        {"icon": "🎯", "title": "PRAGMATIC CONTEXT", "desc": "Intent detection and modality analysis"},
        {"icon": "📊", "title": "ADVANCED METRICS", "desc": "Comprehensive performance evaluation"}
    ]
    
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="netflix-card">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <span style="font-size: 2.5rem; margin-right: 1rem;">{feature['icon']}</span>
                    <h3 style="margin: 0; color: white;">{feature['title']}</h3>
                </div>
                <p style="color: #f5f5f1; margin: 0; line-height: 1.5;">{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

def perform_netflix_analysis(df, config, use_smote):
    """Perform Netflix-style analysis with SMOTE"""
    st.markdown("<div class='section-header'>📈 ANALYSIS RESULTS</div>", unsafe_allow_html=True)
    
    # Data validation
    if config['text_col'] not in df.columns or config['target_col'] not in df.columns:
        st.error("Selected columns not found in dataset.")
        return
    
    if df[config['text_col']].isnull().any():
        df[config['text_col']] = df[config['text_col']].fillna('')
    
    if df[config['target_col']].isnull().any():
        st.error("Target column contains missing values.")
        return
    
    if len(df[config['target_col']].unique()) < 2:
        st.error("Target column must have at least 2 unique classes.")
        return
    
    # Feature extraction
    with st.spinner("🎬 Extracting features..."):
        extractor = NetflixFeatureExtractor()
        X = df[config['text_col']].astype(str)
        y = df[config['target_col']]
        
        if config['feature_type'] == "Lexical":
            X_features = extractor.extract_lexical_features(X)
            feature_desc = "Word-level analysis with lemmatization"
        elif config['feature_type'] == "Semantic":
            X_features = extractor.extract_semantic_features(X)
            feature_desc = "Sentiment analysis and text complexity"
        elif config['feature_type'] == "Syntactic":
            X_features = extractor.extract_syntactic_features(X)
            feature_desc = "Grammar structure and POS analysis"
        else:  # Pragmatic
            X_features = extractor.extract_pragmatic_features(X)
            feature_desc = "Context analysis and intent detection"
    
    st.success(f"✅ Feature extraction completed: {feature_desc}")
    
    # Model training with SMOTE
    with st.spinner("🤖 Training machine learning models with SMOTE..."):
        trainer = NetflixModelTrainer(use_smote=use_smote)
        results, label_encoder = trainer.train_and_evaluate(X_features, y)
    
    # Display results
    successful_models = {k: v for k, v in results.items() if 'error' not in v}
    
    if successful_models:
        # Model Performance Cards
        st.markdown("#### 🎯 MODEL PERFORMANCE")
        
        cols = st.columns(len(successful_models))
        for idx, (model_name, result) in enumerate(successful_models.items()):
            with cols[idx]:
                accuracy = result['accuracy']
                smote_badge = " <span class='smote-badge'>SMOTE</span>" if result.get('smote_applied', False) else ""
                
                st.markdown(f"""
                <div class="model-card">
                    <h4 style="color: white; margin-bottom: 1rem;">{model_name}{smote_badge}</h4>
                    <div class="model-accuracy">{accuracy:.1%}</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; color: #f5f5f1;">
                        <div style="text-align: center;">
                            <small>Precision</small>
                            <div style="font-weight: bold; color: #e50914;">{result['precision']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>Recall</small>
                            <div style="font-weight: bold; color: #e50914;">{result['recall']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>F1-Score</small>
                            <div style="font-weight: bold; color: #e50914;">{result['f1_score']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>Classes</small>
                            <div style="font-weight: bold; color: #e50914;">{result['n_classes']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Netflix Style Dashboard
        st.markdown("#### 📊 PERFORMANCE DASHBOARD")
        viz = NetflixVisualizer()
        dashboard_fig = viz.create_performance_dashboard(successful_models)
        st.pyplot(dashboard_fig)
        
        # Best Model Recommendation
        best_model = max(successful_models.items(), key=lambda x: x[1]['accuracy'])
        smote_status = " (with SMOTE)" if best_model[1].get('smote_applied', False) else ""
        
        st.markdown(f"""
        <div class="netflix-card">
            <h3 style="color: #e50914; margin-bottom: 1rem;">🎬 RECOMMENDED MODEL</h3>
            <p style="color: white; font-size: 1.2rem;">
                <strong>{best_model[0]}</strong>{smote_status} achieved the highest accuracy of 
                <strong style="color: #e50914;">{best_model[1]['accuracy']:.1%}</strong>
            </p>
            <p style="color: #f5f5f1;">
                This model is recommended for deployment based on comprehensive performance metrics.
                { "SMOTE significantly improved performance by handling class imbalance." if best_model[1].get('smote_applied', False) else "" }
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # SMOTE Impact Analysis
        if use_smote and any(result.get('smote_applied', False) for result in successful_models.values()):
            st.markdown("#### 🔧 SMOTE IMPACT ANALYSIS")
            st.info("""
            **SMOTE (Synthetic Minority Over-sampling Technique)** helps improve model performance by:
            - Generating synthetic samples for minority classes
            - Reducing bias towards majority classes
            - Improving recall for underrepresented categories
            - Enhancing overall model generalization
            """)
    
    else:
        st.error("❌ No models were successfully trained. Please check your data and configuration.")

# ============================
# Main Application
# ============================
def main():
    # Initialize session state
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False
    if 'use_smote' not in st.session_state:
        st.session_state.use_smote = True
    
    # Setup sidebar
    setup_sidebar()
    
    # Main content
    main_content()

if __name__ == "__main__":
    main()
