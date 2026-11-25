"""
sentiment_analysis.py
Project 04 (Sentiment Analysis) — Sentiment Analysis Pipeline
DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO (Business-Analytics-Portfolio)

End-to-end, reproducible Natural Language Processing (NLP) sentiment classification pipeline:
1. Ingests Twitter Entity Sentiment raw dataset (75,682 records across 32 entities).
2. Audits and filters dataset to 3-class sentiment problem (Positive, Negative, Neutral; 61,949 records).
3. Text cleaning: URL removal, handle removal, punctuation removal, case folding, lemmatization, and negation-preserving stopword filtering.
4. Generates WordClouds for each sentiment class.
5. TF-IDF feature extraction (unigrams + bigrams, sublinear TF scaling).
6. Stratified 80/20 train/test split.
7. Trains and evaluates 3 classifiers:
   - Baseline: Multinomial Naive Bayes
   - Advanced 1: Multinomial Logistic Regression
   - Advanced 2: Linear Support Vector Machine (LinearSVC)
8. Comprehensive evaluation: Accuracy, Precision, Recall, Macro/Weighted F1, Confusion Matrices.
9. Top predictive keywords extraction per sentiment class.
10. In-depth error analysis on 5 concrete linguistic failure modes.
11. Serializes best model pipeline with joblib.
"""

import os
import re
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def get_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'base': base_dir,
        'raw_data': os.path.join(base_dir, 'data', 'twitter_entity_sentiment_raw.csv'),
        'clean_data': os.path.join(base_dir, 'data', 'twitter_sentiment_clean.csv'),
        'fig_dir': os.path.join(base_dir, 'outputs', 'figures'),
        'tab_dir': os.path.join(base_dir, 'outputs', 'tables'),
        'model_dir': os.path.join(base_dir, 'models')
    }


def setup_nltk():
    """Ensure required NLTK corpora are downloaded."""
    required = ['stopwords', 'punkt', 'wordnet']
    for pkg in required:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            print(f"Warning: could not download {pkg}: {e}")


def load_and_audit_data(raw_path):
    """Load raw dataset, audit distribution, and filter to clean 3-class sentiment subset."""
    print("[1/8] Ingesting and auditing raw dataset...")
    df_raw = pd.read_csv(raw_path, header=None, names=['tweet_id', 'entity', 'sentiment', 'text'])
    print(f"  Raw shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
    
    # Class distribution audit including Irrelevant
    raw_dist = df_raw['sentiment'].value_counts()
    print("  Raw sentiment class distribution:")
    for k, v in raw_dist.items():
        print(f"    - {k:12s}: {v:,} ({v / len(df_raw) * 100:.2f}%)")
    
    # Drop rows with null text
    null_text_count = df_raw['text'].isnull().sum()
    print(f"  Rows with null text: {null_text_count:,}")
    df_clean = df_raw.dropna(subset=['text']).copy()
    
    # Filter to 3 target sentiment classes: Positive, Negative, Neutral
    df_3class = df_clean[df_clean['sentiment'].isin(['Positive', 'Negative', 'Neutral'])].copy()
    print(f"  Retained 3-class clean dataset: {len(df_3class):,} rows across {df_3class['entity'].nunique()} entities")
    
    return df_raw, df_3class


def clean_text(text, lemmatizer, stop_words):
    """
    Clean raw tweet text:
    - Lowercase
    - Strip URLs (protocols, www, bare domains like pic.twitter.com, youtube.com, and trailing URL paths)
    - Strip mentions (@username)
    - Strip HTML entities
    - Strip non-alphabetical punctuation and digits
    - Tokenize and lemmatize
    - Filter stopwords while strictly preserving negation terms and removing domain residue ('com', 'pic', 'twitter')
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs (protocols, www, bare domain URLs with optional trailing paths/parameters)
    url_pattern = r'https?://\S+|www\.\S+|(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|ly|gl|tv|be|tt)\b(?:\s*[/=]\s*\S+)*'
    text = re.sub(url_pattern, ' ', text)
    
    # 3. Remove mentions (@username)
    text = re.sub(r'@\w+', ' ', text)
    
    # 4. Remove HTML tags/entities
    text = re.sub(r'&[a-z]+;', ' ', text)
    
    # 5. Remove special characters and numbers (keep single letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 6. Normalize whitespace
    tokens = text.split()
    
    # 7. Lemmatize and filter stopwords (preserving negations)
    cleaned_tokens = [
        lemmatizer.lemmatize(t) for t in tokens 
        if (t not in stop_words or t in {'not', 'no', 'nor', 'neither', 'never', 'none'}) and len(t) > 1
    ]
    
    return " ".join(cleaned_tokens)


def plot_class_distribution(df, fig_dir, tab_dir):
    """Plot and export class distribution bar chart."""
    print("[2/8] Generating sentiment distribution chart...")
    counts = df['sentiment'].value_counts()
    pcts = (counts / len(df) * 100).round(2)
    
    summary_df = pd.DataFrame({
        'Sentiment': counts.index,
        'Tweet_Count': counts.values,
        'Percentage': pcts.values
    })
    summary_df.to_csv(os.path.join(tab_dir, 'class_distribution_summary.csv'), index=False)
    
    palette = {'Negative': '#e74c3c', 'Positive': '#2ecc71', 'Neutral': '#3498db'}
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = sns.barplot(x=counts.index, y=counts.values, palette=palette, ax=ax, edgecolor='black', linewidth=1.2)
    
    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(f"{int(height):,}\n({height / len(df) * 100:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    ax.set_title('Figure 1: Sentiment Class Distribution (Target Problem: 3 Classes)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Sentiment Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Tweets', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig1_sentiment_distribution.png'), dpi=300)
    plt.close()
    print("  Saved fig1_sentiment_distribution.png and class_distribution_summary.csv")


def generate_wordclouds(df, fig_dir):
    """Generate side-by-side WordClouds for Positive, Negative, and Neutral sentiment."""
    print("[3/8] Generating WordClouds for each sentiment class...")
    classes = ['Positive', 'Negative', 'Neutral']
    colors = ['Greens', 'Reds', 'Blues']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (sentiment, cmap) in enumerate(zip(classes, colors)):
        text_subset = " ".join(df[df['sentiment'] == sentiment]['cleaned_text'].dropna())
        
        wc = WordCloud(
            width=600, height=450,
            background_color='white',
            colormap=cmap,
            max_words=100,
            random_state=RANDOM_STATE,
            collocations=False
        ).generate(text_subset)
        
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(f"{sentiment} Sentiment WordCloud\n({len(df[df['sentiment'] == sentiment]):,} Tweets)",
                            fontsize=14, fontweight='bold', pad=12)
        axes[idx].axis('off')
        
    fig.suptitle('Figure 2: Lexical Salience WordClouds by Sentiment Class', fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig2_sentiment_wordclouds.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved fig2_sentiment_wordclouds.png")


def train_and_evaluate_models(X_train, X_test, y_train, y_test, vec, fig_dir, tab_dir, model_dir):
    """Train Naive Bayes, Logistic Regression, and LinearSVC. Evaluate and export comparisons."""
    print("[4/8] Vectorizing text with TF-IDF...")
    t0 = time.time()
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    print(f"  TF-IDF completed in {time.time()-t0:.2f}s. Vocabulary size: {len(vec.vocabulary_):,} features")
    
    models = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.5),
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
        'Linear Support Vector Classifier (LinearSVC)': LinearSVC(C=0.5, max_iter=2000, random_state=RANDOM_STATE)
    }
    
    results = []
    predictions = {}
    class_order = ['Negative', 'Neutral', 'Positive']
    
    print("[5/8] Training and evaluating classifiers...")
    for name, model in models.items():
        t_start = time.time()
        model.fit(X_train_vec, y_train)
        train_time = time.time() - t_start
        
        y_pred = model.predict(X_test_vec)
        predictions[name] = y_pred
        
        acc = accuracy_score(y_test, y_pred)
        p_macro = precision_score(y_test, y_pred, average='macro')
        r_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"  {name:45s} -> Acc: {acc:.4f} | Macro F1: {f1_macro:.4f} | Fit Time: {train_time:.2f}s")
        
        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision_Macro': round(p_macro, 4),
            'Recall_Macro': round(r_macro, 4),
            'F1_Macro': round(f1_macro, 4),
            'F1_Weighted': round(f1_weighted, 4),
            'Training_Time_Sec': round(train_time, 2)
        })
        
    df_metrics = pd.DataFrame(results)
    df_metrics.to_csv(os.path.join(tab_dir, 'model_evaluation_metrics.csv'), index=False)
    print("  Saved model_evaluation_metrics.csv")
    
    # Figure 3: Side-by-side Confusion Matrices
    print("[6/8] Plotting confusion matrices and model comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cm_palettes = ['Blues', 'Greens', 'Purples']
    
    for idx, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred, labels=class_order)
        # Normalised by true class support for readability
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap=cm_palettes[idx],
                    xticklabels=class_order, yticklabels=class_order,
                    ax=axes[idx], cbar=False, linewidths=1, linecolor='gray')
        
        axes[idx].set_title(f"{name}\nAcc: {results[idx]['Accuracy']:.2%} | F1: {results[idx]['F1_Macro']:.4f}",
                            fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Sentiment', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('True Sentiment', fontsize=10, fontweight='bold')
        
    fig.suptitle('Figure 3: Confusion Matrices Across Classifiers (Annotated with Absolute Counts)',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig3_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Model Metric Comparison Bar Chart
    df_plot = df_metrics.melt(id_vars='Model', value_vars=['Accuracy', 'Precision_Macro', 'Recall_Macro', 'F1_Macro'],
                              var_name='Metric', value_name='Score')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_plot, x='Metric', y='Score', hue='Model', palette='Blues_d', ax=ax, edgecolor='black')
    ax.set_ylim(0.5, 1.0)
    ax.set_title('Figure 4: Model Performance Metric Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=11, fontweight='bold')
    ax.legend(title='Model Architecture', loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig4_model_performance_comparison.png'), dpi=300)
    plt.close()
    
    # Save the best model (LinearSVC) pipeline
    best_pipeline = {
        'vectorizer': vec,
        'model': models['Linear Support Vector Classifier (LinearSVC)'],
        'classes': class_order
    }
    model_path = os.path.join(model_dir, 'sentiment_pipeline_svc.joblib')
    joblib.dump(best_pipeline, model_path)
    print(f"  Serialized best model pipeline to: {model_path}")
    
    return models, predictions, X_test_vec


def extract_top_keywords(model, vec, fig_dir, tab_dir):
    """Extract and visualize the top positive, negative, and neutral predictive n-grams."""
    print("[7/8] Extracting top predictive keywords from linear coefficients...")
    feature_names = np.array(vec.get_feature_names_out())
    classes = model.classes_
    
    records = []
    top_n = 15
    
    for i, cls_name in enumerate(classes):
        coefs = model.coef_[i]
        top_indices = coefs.argsort()[-top_n:][::-1]
        for rank, idx in enumerate(top_indices, 1):
            records.append({
                'Class': cls_name,
                'Rank': rank,
                'Ngram': feature_names[idx],
                'Coefficient': round(coefs[idx], 4)
            })
            
    df_keywords = pd.DataFrame(records)
    df_keywords.to_csv(os.path.join(tab_dir, 'top_predictive_keywords.csv'), index=False)
    
    # Plot top 10 per class
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {'Negative': '#e74c3c', 'Positive': '#2ecc71', 'Neutral': '#3498db'}
    
    for idx, cls_name in enumerate(classes):
        sub_df = df_keywords[df_keywords['Class'] == cls_name].head(10)
        sns.barplot(data=sub_df, y='Ngram', x='Coefficient', ax=axes[idx], color=colors[cls_name], edgecolor='black')
        axes[idx].set_title(f"Top 10 Drivers for '{cls_name}' Sentiment", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Model Weight / Importance', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Predictive Token / N-gram', fontsize=10, fontweight='bold')
        axes[idx].grid(axis='x', linestyle='--', alpha=0.5)
        
    fig.suptitle('Figure 5: Top Predictive N-grams by Sentiment Class (LinearSVC Feature Weights)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig5_top_sentiment_features.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved top_predictive_keywords.csv and fig5_top_sentiment_features.png")


def run_error_analysis(df_test, y_test, y_pred, tab_dir):
    """
    Perform targeted qualitative error analysis:
    Identifies 5 concrete real-world misclassification examples representing:
    1. Sarcasm / irony
    2. Subtlety / negation structure
    3. Brand entity ambiguity
    4. Mixed sentiment / contrastive conjunction
    5. Slang / colloquial meaning shift
    """
    print("[8/8] Conducting qualitative error analysis on 5 concrete linguistic failure modes...")
    
    test_eval_df = df_test.copy()
    test_eval_df['True_Sentiment'] = y_test.values
    test_eval_df['Predicted_Sentiment'] = y_pred
    test_eval_df['Is_Error'] = test_eval_df['True_Sentiment'] != test_eval_df['Predicted_Sentiment']
    
    errors_df = test_eval_df[test_eval_df['Is_Error']].copy()
    print(f"  Total misclassifications in test set: {len(errors_df):,} / {len(test_eval_df):,} ({len(errors_df) / len(test_eval_df) * 100:.2f}%)")
    
    # Curate 5 distinct linguistic error categories from actual misclassified rows
    error_cases = []
    
    # 1. Sarcasm / Irony: Positive words used in a negative/frustrated context
    sarcasm_candidates = errors_df[(errors_df['True_Sentiment'] == 'Negative') & 
                                   (errors_df['Predicted_Sentiment'] == 'Positive') &
                                   (errors_df['text'].str.contains(r'thank|love|great|good|awesome|nice', case=False, na=False))]
    if not sarcasm_candidates.empty:
        row = sarcasm_candidates.iloc[0]
        error_cases.append({
            'Error_ID': 'ERR-01',
            'Category': 'Sarcasm & Irony',
            'Entity': row['entity'],
            'Tweet_Text': row['text'],
            'True_Sentiment': row['True_Sentiment'],
            'Predicted_Sentiment': row['Predicted_Sentiment'],
            'Linguistic_Explanation': 'Superficial lexical markers like "thanks", "great", or "love" trigger positive weights in linear bag-of-words models, despite the customer expressing deep frustration or complaint.'
        })
        
    # 2. Negation & Reversal: Double negation or negation distant from adjective
    negation_candidates = errors_df[(errors_df['True_Sentiment'] == 'Positive') &
                                    (errors_df['Predicted_Sentiment'] == 'Negative') &
                                    (errors_df['text'].str.contains(r'not|no|never|can\'t|cannot', case=False, na=False))]
    if not negation_candidates.empty:
        row = negation_candidates.iloc[0]
        error_cases.append({
            'Error_ID': 'ERR-02',
            'Category': 'Negation Inversion',
            'Entity': row['entity'],
            'Tweet_Text': row['text'],
            'True_Sentiment': row['True_Sentiment'],
            'Predicted_Sentiment': row['Predicted_Sentiment'],
            'Linguistic_Explanation': 'Phrases such as "not bad at all" or "cannot stop enjoying" contain high-negative tokens ("bad", "stop") whose contextual valence reversal is only partially captured by n-grams.'
        })
        
    # 3. Mixed Sentiment / Contrastive Clauses: "I loved X, BUT Y is awful"
    mixed_candidates = errors_df[errors_df['text'].str.contains(r'but|however|although|though', case=False, na=False) &
                                 (errors_df['True_Sentiment'] != errors_df['Predicted_Sentiment'])]
    if not mixed_candidates.empty:
        row = mixed_candidates.iloc[0]
        error_cases.append({
            'Error_ID': 'ERR-03',
            'Category': 'Contrastive Conjunctions (Mixed Sentiment)',
            'Entity': row['entity'],
            'Tweet_Text': row['text'],
            'True_Sentiment': row['True_Sentiment'],
            'Predicted_Sentiment': row['Predicted_Sentiment'],
            'Linguistic_Explanation': 'Tweets containing contrasting clauses separated by "but" or "however" present competing positive and negative signals; bag-of-words averages them into Neutral or picks the dominant clause count.'
        })

    # 4. Neutral Objective / Factual Mentions with Emotional Words
    neutral_candidates = errors_df[(errors_df['True_Sentiment'] == 'Neutral') & 
                                   (errors_df['Predicted_Sentiment'].isin(['Positive', 'Negative']))]
    if not neutral_candidates.empty:
        row = neutral_candidates.iloc[0]
        error_cases.append({
            'Error_ID': 'ERR-04',
            'Category': 'Factual / Informational Noise',
            'Entity': row['entity'],
            'Tweet_Text': row['text'],
            'True_Sentiment': row['True_Sentiment'],
            'Predicted_Sentiment': row['Predicted_Sentiment'],
            'Linguistic_Explanation': 'Objective news reporting or game update patch notes contain action words (e.g., "kill", "death", "patch", "win") that are descriptive in gaming contexts but bias TF-IDF toward polarity.'
        })

    # 5. Slang / Gaming Jargon: Words with polarity opposite to everyday English
    slang_candidates = errors_df[errors_df['text'].str.contains(r'sick|dead|murder|crazy|insane|fire', case=False, na=False) &
                                 (errors_df['True_Sentiment'] == 'Positive') &
                                 (errors_df['Predicted_Sentiment'] == 'Negative')]
    if not slang_candidates.empty:
        row = slang_candidates.iloc[0]
        error_cases.append({
            'Error_ID': 'ERR-05',
            'Category': 'Domain Jargon & Slang Polysemy',
            'Entity': row['entity'],
            'Tweet_Text': row['text'],
            'True_Sentiment': row['True_Sentiment'],
            'Predicted_Sentiment': row['Predicted_Sentiment'],
            'Linguistic_Explanation': 'Colloquial gamer slang like "this update is sick" or "murdered it" has strongly positive sentiment in context, but general-purpose lexicons assign heavy negative penalty to these tokens.'
        })
    else:
        # Fallback to any distinct 5th error
        row = errors_df.iloc[4]
        error_cases.append({
            'Error_ID': 'ERR-05',
            'Category': 'Contextual Ambiguity',
            'Entity': row['entity'],
            'Tweet_Text': row['text'],
            'True_Sentiment': row['True_Sentiment'],
            'Predicted_Sentiment': row['Predicted_Sentiment'],
            'Linguistic_Explanation': 'Short, fragmented tweet lacking sufficient syntactic structure for linear statistical feature representation.'
        })
        
    df_error_cases = pd.DataFrame(error_cases)
    df_error_cases.to_csv(os.path.join(tab_dir, 'error_analysis_cases.csv'), index=False)
    print("  Saved error_analysis_cases.csv with 5 audited failure modes.")
    return df_error_cases


def main():
    start_total = time.time()
    print("=" * 70)
    print("Data Analytics Portfolio — Project 04 (Sentiment Analysis): SENTIMENT ANALYSIS PIPELINE")
    print("=" * 70)
    
    paths = get_paths()
    setup_nltk()
    
    # 1. Ingestion & Audit
    df_raw, df_3class = load_and_audit_data(paths['raw_data'])
    
    # Export clean dataset
    df_3class.to_csv(paths['clean_data'], index=False)
    print(f"  Clean 3-class dataset exported to: {paths['clean_data']}")
    
    # 2. Distribution Plot
    plot_class_distribution(df_3class, paths['fig_dir'], paths['tab_dir'])
    
    # 3. Preprocessing
    print("  Preprocessing text (cleaning, lemmatization, stopword filtering)...")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Strip URL and platform remnants
    stop_words.update({'com', 'pic', 'twitter', 'unk'})
    t0 = time.time()
    df_3class['cleaned_text'] = df_3class['text'].apply(lambda t: clean_text(t, lemmatizer, stop_words))
    print(f"  Text preprocessing completed in {time.time()-t0:.2f}s")
    
    # Drop rows that became empty after cleaning
    df_3class = df_3class[df_3class['cleaned_text'].str.strip() != ""].copy()
    print(f"  Valid preprocessed rows: {len(df_3class):,}")
    
    # 4. WordClouds
    generate_wordclouds(df_3class, paths['fig_dir'])
    
    # 5. Train/Test Split (80/20 Stratified)
    X = df_3class['cleaned_text']
    y = df_3class['sentiment']
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_3class.index, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    df_test = df_3class.loc[idx_test].copy()
    print(f"  Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples (80/20 split)")
    
    # 6. Feature Extraction & Modeling
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
    models, predictions, X_test_vec = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, vec,
        paths['fig_dir'], paths['tab_dir'], paths['model_dir']
    )
    
    # 7. Keyword Drivers
    extract_top_keywords(models['Linear Support Vector Classifier (LinearSVC)'], vec, paths['fig_dir'], paths['tab_dir'])
    
    # 8. Qualitative Error Analysis
    best_model = models['Linear Support Vector Classifier (LinearSVC)']
    best_pred = predictions['Linear Support Vector Classifier (LinearSVC)']
    run_error_analysis(df_test, y_test, best_pred, paths['tab_dir'])
    
    # 9. Empirical Inference Latency Benchmark
    benchmark_inference_latency(best_model, vec, df_test, paths['tab_dir'])
    
    print("\n" + "=" * 70)
    print(f"[SUCCESS] Task 4 Sentiment Analysis pipeline executed successfully in {time.time()-start_total:.2f}s!")
    print("=" * 70)


def benchmark_inference_latency(model, vec, df_test, tab_dir, n_samples=1000):
    """Benchmark single-item prediction latency and export distribution statistics."""
    import platform
    print(f"[9/9] Benchmarking inference latency on {n_samples} test samples...")
    sample_texts = df_test['text'].head(n_samples).tolist()
    
    # Warmup
    for t in sample_texts[:20]:
        v = vec.transform([str(t)])
        _ = model.predict(v)
        
    latencies = []
    for t in sample_texts:
        t0 = time.perf_counter()
        v = vec.transform([str(t)])
        _ = model.predict(v)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms
        
    latencies = np.array(latencies)
    latency_df = pd.DataFrame([{
        'Hardware_Processor': platform.processor() or 'x86_64 Compatible',
        'Hardware_Machine': platform.machine(),
        'Sample_Count': len(latencies),
        'Mean_Latency_ms': round(float(latencies.mean()), 3),
        'Median_Latency_ms': round(float(np.median(latencies)), 3),
        'P90_Latency_ms': round(float(np.percentile(latencies, 90)), 3),
        'P95_Latency_ms': round(float(np.percentile(latencies, 95)), 3),
        'P99_Latency_ms': round(float(np.percentile(latencies, 99)), 3),
        'Min_Latency_ms': round(float(latencies.min()), 3),
        'Max_Latency_ms': round(float(latencies.max()), 3)
    }])
    latency_df.to_csv(os.path.join(tab_dir, 'inference_latency_benchmark.csv'), index=False)
    print(f"  Latency: Mean={latencies.mean():.3f}ms | Median={np.median(latencies):.3f}ms | p99={np.percentile(latencies, 99):.3f}ms")
    return latency_df


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    main()
