"""
FINAL NLP PROJECT: Sarcasm Detection Robustness Analysis
Comparing Traditional ML, Zero-Shot LLM, Few-Shot LLM, and Fine-Tuned Models

"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from datasets import load_dataset
import anthropic
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR SPEED AND COST
# ============================================================================

CONFIG = {
    'model': 'claude-3-5-haiku-20241022',  
    'sample_size': 100,  # Reduced for speed (still statistically valid)
    'train_samples': 3000,  # Reduced training data for faster fine-tuning
    'epochs': 2,
    'batch_size': 16
}

print(f"""
{'='*60}
OPTIMIZED CONFIGURATION
{'='*60}
Model: {CONFIG['model']}
Sample Size: {CONFIG['sample_size']} 
Estimated Cost: ~$0.08 total
Estimated API Time: ~15-20 minutes
Training Time: ~15-20 minutes
{'='*60}
""")

# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_sarcasm_data():
    
    print("Loading dataset...")
    # Using the news headlines sarcasm dataset
    dataset = load_dataset("raquiba/Sarcasm_News_Headline")
    
    # Convert to pandas for easier manipulation
    # NOTE: The dataset has unusual splits (52% train, 48% test)
    # We combine them and create a proper 80/20 split
    all_data = pd.concat([
        pd.DataFrame(dataset['train']),
        pd.DataFrame(dataset['test'])
    ], ignore_index=True)
    
    # Rename columns for clarity
    all_data = all_data.rename(columns={'headline': 'text', 'is_sarcastic': 'label'})
    
    # Create proper train/test split (80/20)
    train_df, test_df = train_test_split(
        all_data, 
        test_size=0.2, 
        random_state=42, 
        stratify=all_data['label']  # Maintain class balance
    )
    
    print(f"Total samples: {len(all_data)}")
    print(f"Train size: {len(train_df)} (80%), Test size: {len(test_df)} (20%)")
    print(f"Sarcastic samples in train: {train_df['label'].sum()} ({train_df['label'].mean()*100:.1f}%)")
    print(f"Sarcastic samples in test: {test_df['label'].sum()} ({test_df['label'].mean()*100:.1f}%)")
    
    return train_df, test_df

# ============================================================================
# SECTION 2: METHOD 1 - TRADITIONAL ML BASELINE (TF-IDF + Logistic Regression)
# ============================================================================

def train_traditional_baseline(train_df, test_df, max_samples=3000):
    """Train traditional ML baseline using TF-IDF + Logistic Regression"""
    print("\n" + "="*50)
    print("METHOD 1: TRADITIONAL ML BASELINE")
    print("="*50)
    
    # Sample training data for fair comparison
    train_subset = train_df.sample(n=min(max_samples, len(train_df)), random_state=42)
    
    print(f"Training traditional ML model on {len(train_subset)} samples")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=2,
            stop_words='english'
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # Train
    print("Training TF-IDF + Logistic Regression...")
    pipeline.fit(train_subset['text'], train_subset['label'])
    
    # Predict on full test set
    print("Evaluating on test set...")
    predictions = pipeline.predict(test_df['text'])
    probs = pipeline.predict_proba(test_df['text'])
    
    # Calculate metrics
    acc = accuracy_score(test_df['label'], predictions)
    p, r, f1, _ = precision_recall_fscore_support(test_df['label'], predictions, average='binary')
    
    print(f"\nTraditional ML Results:")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {p:.3f}")
    print(f"Recall: {r:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # Get feature importance (top predictive words)
    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    coefficients = pipeline.named_steps['classifier'].coef_[0]
    
    # Top features for sarcasm
    top_sarcasm_idx = np.argsort(coefficients)[-10:]
    top_sarcasm_features = [(feature_names[i], coefficients[i]) for i in top_sarcasm_idx]
    
    # Top features for not sarcasm
    top_not_sarcasm_idx = np.argsort(coefficients)[:10]
    top_not_sarcasm_features = [(feature_names[i], coefficients[i]) for i in top_not_sarcasm_idx]
    
    print("\nTop 10 features predicting SARCASM:")
    for feature, coef in reversed(top_sarcasm_features):
        print(f"  {feature}: {coef:.3f}")
    
    print("\nTop 10 features predicting NOT SARCASM:")
    for feature, coef in top_not_sarcasm_features:
        print(f"  {feature}: {coef:.3f}")
    
    return pipeline, predictions, probs

# ============================================================================
# SECTION 3: METHOD 2 - ZERO-SHOT LLM
# ============================================================================

def zero_shot_predict(text, client):
    """Zero-shot sarcasm detection using Claude"""
    prompt = f"""Is the following headline sarcastic or not sarcastic?

Headline: "{text}"

Respond with ONLY one word: either "sarcastic" or "not_sarcastic"."""
    
    try:
        message = client.messages.create(
            model=CONFIG['model'],
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content[0].text.strip().lower()
        
        # Parse response
        if "sarcastic" in response and "not" not in response:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"Error in zero-shot: {e}")
        return 0

def evaluate_zero_shot(test_df, client, sample_size=None):
    """Evaluate zero-shot approach"""
    if sample_size is None:
        sample_size = CONFIG['sample_size']
    
    print("\n" + "="*50)
    print("METHOD 2: ZERO-SHOT LLM")
    print("="*50)
    print(f"Using {sample_size} samples")
    
    # Sample for faster evaluation
    test_sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
    
    predictions = []
    for idx, row in test_sample.iterrows():
        pred = zero_shot_predict(row['text'], client)
        predictions.append(pred)
        if len(predictions) % 20 == 0:
            print(f"Processed {len(predictions)}/{len(test_sample)} samples...")
    
    test_sample = test_sample.copy()
    test_sample['zero_shot_pred'] = predictions
    
    # Calculate metrics
    acc = accuracy_score(test_sample['label'], predictions)
    p, r, f1, _ = precision_recall_fscore_support(test_sample['label'], predictions, average='binary')
    
    print(f"\nZero-Shot LLM Results:")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {p:.3f}")
    print(f"Recall: {r:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    return test_sample, predictions

# ============================================================================
# SECTION 4: METHOD 3 - FEW-SHOT LLM
# ============================================================================

def get_few_shot_examples(train_df, n_examples=5):
    """Get balanced few-shot examples"""
    sarcastic = train_df[train_df['label'] == 1].sample(n=n_examples//2 + 1, random_state=42)
    not_sarcastic = train_df[train_df['label'] == 0].sample(n=n_examples//2, random_state=42)
    examples = pd.concat([sarcastic, not_sarcastic]).sample(frac=1, random_state=42)
    return examples

def few_shot_predict(text, examples, client):
    """Few-shot sarcasm detection using Claude"""
    
    # Build prompt with examples
    prompt = "Here are some examples of sarcastic and non-sarcastic headlines:\n\n"
    
    for _, row in examples.iterrows():
        label_text = "sarcastic" if row['label'] == 1 else "not_sarcastic"
        prompt += f'Headline: "{row["text"]}"\nLabel: {label_text}\n\n'
    
    prompt += f'Now classify this headline:\n\nHeadline: "{text}"\n\nRespond with ONLY one word: either "sarcastic" or "not_sarcastic".'
    
    try:
        message = client.messages.create(
            model=CONFIG['model'],
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content[0].text.strip().lower()
        
        if "sarcastic" in response and "not" not in response:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"Error in few-shot: {e}")
        return 0

def evaluate_few_shot(test_sample, train_df, client):
    """Evaluate few-shot approach"""
    print("\n" + "="*50)
    print("METHOD 3: FEW-SHOT LLM")
    print("="*50)
    
    examples = get_few_shot_examples(train_df, n_examples=5)
    print(f"Using {len(examples)} few-shot examples")
    
    predictions = []
    for idx, row in test_sample.iterrows():
        pred = few_shot_predict(row['text'], examples, client)
        predictions.append(pred)
        if len(predictions) % 20 == 0:
            print(f"Processed {len(predictions)}/{len(test_sample)} samples...")
    
    test_sample = test_sample.copy()
    test_sample['few_shot_pred'] = predictions
    
    # Calculate metrics
    acc = accuracy_score(test_sample['label'], predictions)
    p, r, f1, _ = precision_recall_fscore_support(test_sample['label'], predictions, average='binary')
    
    print(f"\nFew-Shot LLM Results:")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {p:.3f}")
    print(f"Recall: {r:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    return test_sample, predictions

# ============================================================================
# SECTION 5: METHOD 4 - FINE-TUNED TRANSFORMER
# ============================================================================

def prepare_dataset_for_training(train_df, test_df, tokenizer, max_samples=None):
    """Prepare dataset for fine-tuning"""
    if max_samples is None:
        max_samples = CONFIG['train_samples']
    
    # Sample for faster training
    train_subset = train_df.sample(n=min(max_samples, len(train_df)), random_state=42)
    
    print(f"Training on {len(train_subset)} samples")
    
    # Tokenize
    train_encodings = tokenizer(train_subset['text'].tolist(), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=128)
    
    # Create torch datasets
    class SarcasmDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        
        def __len__(self):
            return len(self.labels)
    
    train_dataset = SarcasmDataset(train_encodings, train_subset['label'].tolist())
    test_dataset = SarcasmDataset(test_encodings, test_df['label'].tolist())
    
    return train_dataset, test_dataset

def train_fine_tuned_model(train_df, test_df):
    """Fine-tune a model for sarcasm detection"""
    print("\n" + "="*50)
    print("METHOD 4: FINE-TUNED TRANSFORMER")
    print("="*50)
    
    # Load model and tokenizer
    model_name = "distilbert-base-uncased"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Prepare datasets
    train_dataset, test_dataset = prepare_dataset_for_training(train_df, test_df, tokenizer)
    
    # Training arguments 
    training_args = TrainingArguments(
        output_dir="./sarcasm_model",
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Training model...")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating fine-tuned model...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    
    acc = accuracy_score(test_df['label'], preds)
    p, r, f1, _ = precision_recall_fscore_support(test_df['label'], preds, average='binary')
    
    print(f"\nFine-Tuned Transformer Results:")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {p:.3f}")
    print(f"Recall: {r:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    return model, tokenizer, preds

# ============================================================================
# SECTION 6: COMPREHENSIVE ERROR ANALYSIS
# ============================================================================

def comprehensive_error_analysis(test_sample, test_df, trad_preds, ft_preds):
    """Perform detailed error analysis across all methods"""
    print("\n" + "="*50)
    print("COMPREHENSIVE ERROR ANALYSIS")
    print("="*50)
    
    # Add traditional ML predictions to test_sample
    # Match indices
    test_sample = test_sample.copy()
    test_sample['trad_ml_pred'] = test_sample.index.map(
        lambda idx: trad_preds[test_df.index.get_loc(idx)]
    )
    
    # Identify errors for each method
    test_sample['trad_ml_error'] = test_sample['label'] != test_sample['trad_ml_pred']
    test_sample['zero_shot_error'] = test_sample['label'] != test_sample['zero_shot_pred']
    test_sample['few_shot_error'] = test_sample['label'] != test_sample['few_shot_pred']
    
    # Categorize error types
    def categorize_text(text):
        """Categorize text by content type"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['trump', 'obama', 'biden', 'politics', 'congress', 'senate', 'president']):
            return 'political'
        elif any(word in text_lower for word in ['win', 'lose', 'game', 'team', 'sport', 'nba', 'nfl', 'baseball']):
            return 'sports'
        elif any(word in text_lower for word in ['study', 'research', 'scientist', 'finds', 'science', 'university']):
            return 'science'
        elif len(text.split()) < 8:
            return 'short'
        elif len(text.split()) > 15:
            return 'long'
        else:
            return 'general'
    
    test_sample['category'] = test_sample['text'].apply(categorize_text)
    
    # Analysis by category
    print("\n" + "="*50)
    print("ERROR RATES BY CATEGORY")
    print("="*50)
    
    category_stats = []
    for cat in sorted(test_sample['category'].unique()):
        cat_data = test_sample[test_sample['category'] == cat]
        stats = {
            'category': cat,
            'count': len(cat_data),
            'trad_ml_error': cat_data['trad_ml_error'].mean(),
            'zero_shot_error': cat_data['zero_shot_error'].mean(),
            'few_shot_error': cat_data['few_shot_error'].mean()
        }
        category_stats.append(stats)
        
        print(f"\n{cat.upper()} ({len(cat_data)} samples):")
        print(f"  Traditional ML: {stats['trad_ml_error']:.2%}")
        print(f"  Zero-Shot LLM:  {stats['zero_shot_error']:.2%}")
        print(f"  Few-Shot LLM:   {stats['few_shot_error']:.2%}")
    
    # Agreement analysis
    print("\n" + "="*50)
    print("MODEL AGREEMENT ANALYSIS")
    print("="*50)
    
    # All models agree and correct
    all_correct = (
        (test_sample['trad_ml_pred'] == test_sample['label']) &
        (test_sample['zero_shot_pred'] == test_sample['label']) &
        (test_sample['few_shot_pred'] == test_sample['label'])
    )
    
    # All models agree but wrong
    all_wrong = (
        (test_sample['trad_ml_pred'] == test_sample['zero_shot_pred']) &
        (test_sample['zero_shot_pred'] == test_sample['few_shot_pred']) &
        (test_sample['trad_ml_pred'] != test_sample['label'])
    )
    
    print(f"All models correct: {all_correct.sum()} ({all_correct.mean():.1%})")
    print(f"All models wrong (agree): {all_wrong.sum()} ({all_wrong.mean():.1%})")
    print(f"Models disagree: {(~all_correct & ~all_wrong).sum()} ({(~all_correct & ~all_wrong).mean():.1%})")
    
    # Sample errors
    print("\n" + "="*50)
    print("EXAMPLE ERRORS")
    print("="*50)
    
    print("\nFalse Positives (All models predicted sarcastic, actually not):")
    all_fp = test_sample[
        (test_sample['label'] == 0) &
        (test_sample['trad_ml_pred'] == 1) &
        (test_sample['zero_shot_pred'] == 1) &
        (test_sample['few_shot_pred'] == 1)
    ].head(3)
    for _, row in all_fp.iterrows():
        print(f"  - {row['text']}")
    
    print("\nFalse Negatives (All models predicted not sarcastic, actually sarcastic):")
    all_fn = test_sample[
        (test_sample['label'] == 1) &
        (test_sample['trad_ml_pred'] == 0) &
        (test_sample['zero_shot_pred'] == 0) &
        (test_sample['few_shot_pred'] == 0)
    ].head(3)
    for _, row in all_fn.iterrows():
        print(f"  - {row['text']}")
    
    print("\nTraditional ML correct, LLMs wrong:")
    trad_right_llm_wrong = test_sample[
        (test_sample['trad_ml_pred'] == test_sample['label']) &
        (test_sample['zero_shot_pred'] != test_sample['label']) &
        (test_sample['few_shot_pred'] != test_sample['label'])
    ].head(3)
    for _, row in trad_right_llm_wrong.iterrows():
        print(f"  - {row['text']}")
    
    print("\nLLMs correct, Traditional ML wrong:")
    llm_right_trad_wrong = test_sample[
        (test_sample['trad_ml_pred'] != test_sample['label']) &
        (test_sample['zero_shot_pred'] == test_sample['label']) &
        (test_sample['few_shot_pred'] == test_sample['label'])
    ].head(3)
    for _, row in llm_right_trad_wrong.iterrows():
        print(f"  - {row['text']}")
    
    return test_sample, pd.DataFrame(category_stats)

# ============================================================================
# SECTION 7: COMPREHENSIVE VISUALIZATION
# ============================================================================

def create_comprehensive_visualizations(test_sample, all_results, category_stats):
    """Create detailed comparison visualizations"""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    sns.set_style("whitegrid")
    
    # 1. Overall Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1a. Accuracy and F1 comparison
    methods = ['Traditional ML', 'Zero-Shot LLM', 'Few-Shot LLM', 'Fine-Tuned']
    accuracies = [
        all_results['traditional']['accuracy'],
        all_results['zero_shot']['accuracy'], 
        all_results['few_shot']['accuracy'],
        all_results['fine_tuned']['accuracy']
    ]
    f1_scores = [
        all_results['traditional']['f1'],
        all_results['zero_shot']['f1'], 
        all_results['few_shot']['f1'],
        all_results['fine_tuned']['f1']
    ]
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue', edgecolor='navy')
    axes[0, 0].bar(x + width/2, f1_scores, width, label='F1-Score', color='lightcoral', edgecolor='darkred')
    axes[0, 0].set_ylabel('Score', fontsize=12)
    axes[0, 0].set_title('Performance Comparison Across Methods', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(methods, rotation=15, ha='right')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
        axes[0, 0].text(i - width/2, acc + 0.02, f'{acc:.2f}', ha='center', fontsize=9)
        axes[0, 0].text(i + width/2, f1 + 0.02, f'{f1:.2f}', ha='center', fontsize=9)
    
    # 1b. Precision-Recall comparison
    precisions = [
        all_results['traditional']['precision'],
        all_results['zero_shot']['precision'],
        all_results['few_shot']['precision'],
        all_results['fine_tuned']['precision']
    ]
    recalls = [
        all_results['traditional']['recall'],
        all_results['zero_shot']['recall'],
        all_results['few_shot']['recall'],
        all_results['fine_tuned']['recall']
    ]
    
    axes[0, 1].scatter(recalls, precisions, s=200, alpha=0.6, c=['green', 'blue', 'orange', 'red'])
    for i, method in enumerate(methods):
        axes[0, 1].annotate(method, (recalls[i], precisions[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[0, 1].set_xlabel('Recall', fontsize=12)
    axes[0, 1].set_ylabel('Precision', fontsize=12)
    axes[0, 1].set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0.5, 1.0])
    axes[0, 1].set_ylim([0.5, 1.0])
    
    # 1c. Error rates by category
    categories = category_stats['category'].tolist()
    x_cat = np.arange(len(categories))
    width = 0.2
    
    axes[1, 0].bar(x_cat - 1.5*width, category_stats['trad_ml_error'], width, 
                   label='Traditional ML', color='green', alpha=0.7)
    axes[1, 0].bar(x_cat - 0.5*width, category_stats['zero_shot_error'], width, 
                   label='Zero-Shot', color='blue', alpha=0.7)
    axes[1, 0].bar(x_cat + 0.5*width, category_stats['few_shot_error'], width, 
                   label='Few-Shot', color='orange', alpha=0.7)
    
    axes[1, 0].set_ylabel('Error Rate', fontsize=12)
    axes[1, 0].set_title('Error Rate by Text Category', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x_cat)
    axes[1, 0].set_xticklabels(categories, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 0.6])
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 1d. Confusion matrix for best model
    cm = confusion_matrix(test_sample['label'], test_sample['few_shot_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], 
                cbar_kws={'label': 'Count'})
    axes[1, 1].set_title('Confusion Matrix (Few-Shot LLM)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('True Label', fontsize=12)
    axes[1, 1].set_xlabel('Predicted Label', fontsize=12)
    axes[1, 1].set_xticklabels(['Not Sarcastic', 'Sarcastic'])
    axes[1, 1].set_yticklabels(['Not Sarcastic', 'Sarcastic'])
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comprehensive_analysis.png")
    
    plt.close('all')

# ============================================================================
# SECTION 8: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*60)
    print("SATIRICAL NEWS HEADLINE DETECTION: COMPREHENSIVE ROBUSTNESS ANALYSIS")
    print("Comparing Traditional ML, LLMs, and Fine-Tuned Models")
    print("="*60)
    
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: Please set ANTHROPIC_API_KEY environment variable")
        print("export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Load data
    train_df, test_df = load_sarcasm_data()
    
    # METHOD 1: Traditional ML Baseline
    trad_pipeline, trad_preds, trad_probs = train_traditional_baseline(train_df, test_df)
    trad_metrics = {
        'accuracy': accuracy_score(test_df['label'], trad_preds),
        'precision': precision_recall_fscore_support(test_df['label'], trad_preds, average='binary')[0],
        'recall': precision_recall_fscore_support(test_df['label'], trad_preds, average='binary')[1],
        'f1': precision_recall_fscore_support(test_df['label'], trad_preds, average='binary')[2]
    }
    
    # METHOD 2: Zero-Shot LLM
    test_sample, zero_shot_preds = evaluate_zero_shot(test_df, client)
    zero_shot_metrics = {
        'accuracy': accuracy_score(test_sample['label'], zero_shot_preds),
        'precision': precision_recall_fscore_support(test_sample['label'], zero_shot_preds, average='binary')[0],
        'recall': precision_recall_fscore_support(test_sample['label'], zero_shot_preds, average='binary')[1],
        'f1': precision_recall_fscore_support(test_sample['label'], zero_shot_preds, average='binary')[2]
    }
    
    # METHOD 3: Few-Shot LLM
    test_sample, few_shot_preds = evaluate_few_shot(test_sample, train_df, client)
    few_shot_metrics = {
        'accuracy': accuracy_score(test_sample['label'], few_shot_preds),
        'precision': precision_recall_fscore_support(test_sample['label'], few_shot_preds, average='binary')[0],
        'recall': precision_recall_fscore_support(test_sample['label'], few_shot_preds, average='binary')[1],
        'f1': precision_recall_fscore_support(test_sample['label'], few_shot_preds, average='binary')[2]
    }
    
    # METHOD 4: Fine-Tuned Transformer
    model, tokenizer, fine_tuned_preds = train_fine_tuned_model(train_df, test_df)
    fine_tuned_metrics = {
        'accuracy': accuracy_score(test_df['label'], fine_tuned_preds),
        'precision': precision_recall_fscore_support(test_df['label'], fine_tuned_preds, average='binary')[0],
        'recall': precision_recall_fscore_support(test_df['label'], fine_tuned_preds, average='binary')[1],
        'f1': precision_recall_fscore_support(test_df['label'], fine_tuned_preds, average='binary')[2]
    }
    
    # Aggregate results
    all_results = {
        'traditional': trad_metrics,
        'zero_shot': zero_shot_metrics,
        'few_shot': few_shot_metrics,
        'fine_tuned': fine_tuned_metrics
    }
    
    # Comprehensive Error Analysis
    test_sample, category_stats = comprehensive_error_analysis(
        test_sample, test_df, trad_preds, fine_tuned_preds
    )
    
    # Visualizations
    create_comprehensive_visualizations(test_sample, all_results, category_stats)
    
    # Save comprehensive results
    results_summary = {
        'Method': ['Traditional ML (TF-IDF+LR)', 'Zero-Shot LLM', 'Few-Shot LLM', 'Fine-Tuned Transformer'],
        'Accuracy': [f"{all_results['traditional']['accuracy']:.3f}",
                    f"{all_results['zero_shot']['accuracy']:.3f}", 
                    f"{all_results['few_shot']['accuracy']:.3f}",
                    f"{all_results['fine_tuned']['accuracy']:.3f}"],
        'Precision': [f"{all_results['traditional']['precision']:.3f}",
                     f"{all_results['zero_shot']['precision']:.3f}", 
                     f"{all_results['few_shot']['precision']:.3f}",
                     f"{all_results['fine_tuned']['precision']:.3f}"],
        'Recall': [f"{all_results['traditional']['recall']:.3f}",
                  f"{all_results['zero_shot']['recall']:.3f}", 
                  f"{all_results['few_shot']['recall']:.3f}",
                  f"{all_results['fine_tuned']['recall']:.3f}"],
        'F1-Score': [f"{all_results['traditional']['f1']:.3f}",
                    f"{all_results['zero_shot']['f1']:.3f}", 
                    f"{all_results['few_shot']['f1']:.3f}",
                    f"{all_results['fine_tuned']['f1']:.3f}"]
    }
    
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv('results_summary.csv', index=False)
    print("\n✓ Saved: results_summary.csv")
    
    # Save category stats
    category_stats.to_csv('error_by_category.csv', index=False)
    print("✓ Saved: error_by_category.csv")
    
    # Save test sample with all predictions
    test_sample.to_csv('test_predictions.csv', index=False)
    print("✓ Saved: test_predictions.csv")
    
    # Save traditional ML model
    import joblib
    joblib.dump(trad_pipeline, 'traditional_ml_model.pkl')
    print(" Saved: traditional_ml_model.pkl")
    
    print("\n" + "="*60)
    print("PROJECT COMPLETE!")
    print("="*60)
    print("\n  Generated Files:")
    print("  1. comprehensive_analysis.png - All visualizations")
    print("  2. results_summary.csv - Complete metrics table")
    print("  3. error_by_category.csv - Category-level statistics")
    print("  4. test_predictions.csv - All model predictions")
    print("  5. traditional_ml_model.pkl - Trained TF-IDF model")
    print("  6. ./sarcasm_model/ - Fine-tuned transformer")
    
    print("\n Summary of Results:")
    print(results_df.to_string(index=False))
    
    print("\n Key Findings:")
    print(f"  • Traditional ML (TF-IDF): {all_results['traditional']['accuracy']:.1%} accuracy")
    print(f"  • Zero-Shot LLM: {all_results['zero_shot']['accuracy']:.1%} accuracy (+{(all_results['zero_shot']['accuracy']-all_results['traditional']['accuracy'])*100:.1f}%)")
    print(f"  • Few-Shot LLM: {all_results['few_shot']['accuracy']:.1%} accuracy (+{(all_results['few_shot']['accuracy']-all_results['traditional']['accuracy'])*100:.1f}%)")
    print(f"  • Fine-Tuned: {all_results['fine_tuned']['accuracy']:.1%} accuracy (+{(all_results['fine_tuned']['accuracy']-all_results['traditional']['accuracy'])*100:.1f}%)")
    
    
    
    return model, tokenizer, test_sample, all_results, trad_pipeline

if __name__ == "__main__":
    model, tokenizer, test_sample, all_results, trad_pipeline = main()