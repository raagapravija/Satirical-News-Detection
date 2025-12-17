# Satirical News Detection: Comparative Analysis of Four NLP Approaches

Comparing Traditional ML, Zero-Shot, Few-Shot, and Fine-Tuned approaches for detecting satirical news headlines.

## Overview

This project systematically compares four distinct NLP paradigms for distinguishing satirical news from real news. Results reveal a fundamental gap: zero-shot LLMs detect linguistic sarcasm (74% accuracy), while fine-tuned models learn source-specific patterns (88% accuracy).

### Results Summary

| Method | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Traditional ML | 72.0% | 69.9% | 68.5% | 69.2% |
| Zero-Shot LLM | 74.0% | 79.5% | 63.3% | 70.5% |
| Few-Shot LLM | 79.0% | 79.2% | 77.6% | 78.4% |
| Fine-Tuned Transformer | 88.0% | 88.2% | 85.3% | 86.7% |

---

## Quick Start

### Prerequisites
- Python 3.8+
- 8GB+ RAM
- Anthropic API key 

### Installation

**Step 1: Setup Virtual Environment**

macOS/Linux:
```bash
mkdir satirical-news-detection && cd satirical-news-detection
python3 -m venv venv
source venv/bin/activate
```

Windows:
```bash
mkdir satirical-news-detection && cd satirical-news-detection
python -m venv venv
venv\Scripts\activate
```

**Step 2: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 3: Set API Key**

macOS/Linux:
```bash
export ANTHROPIC_API_KEY='api-key'
```

Windows:
```bash
set ANTHROPIC_API_KEY= api-key
```

**Step 4: Run Analysis**

```bash
python sarcasm_project.py
```

**Step 5: Launch Web Interface**

```bash
python interface.py
```

Open browser to `http://127.0.0.1:7860`

---

## Project Structure

```
satirical-news-detection/
├── sarcasm_project.py          # Main analysis script
├── interface.py                # Gradio web interface
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── comprehensive_analysis.png  # Visualizations
├── results_summary.csv         # Performance metrics
├── error_by_category.csv       # Error analysis
├── test_predictions.csv        # All predictions
├── traditional_ml_model.pkl    # Trained TF-IDF model
├── sarcasm_model/              # Fine-tuned DistilBERT
└── logs/                       # Training logs
```

---

## Methods Compared

### 1. Traditional ML (TF-IDF + Logistic Regression)
- Training data: 3,000 samples
- Execution: <1 minute
- Strengths: Fast, interpretable, learns keyword patterns
- Limitations: Cannot generalize to novel vocabulary

### 2. Zero-Shot LLM (Claude Haiku)
- No training required
- Prompt: "Is this headline sarcastic?"
- Strengths: No setup needed, detects linguistic sarcasm
- Limitations: Conservative predictions, source-blind

### 3. Few-Shot LLM (Claude Haiku + 5 Examples)
- 5 in-context examples provided
- Improvements: +5% accuracy over zero-shot
- Strengths: Best ROI for practical deployment
- Limitations: Slower inference than fine-tuned

### 4. Fine-Tuned Transformer (DistilBERT)
- Training: 2 epochs on 3,000 samples
- Time: ~20 minutes (GPU), ~2 hours (CPU)
- Strengths: Best performance, learns both linguistic and source patterns
- Limitations: Requires training data and compute

---

## Category-Specific Performance

| Category | Trad. ML | Zero-Shot | Few-Shot | Fine-Tuned |
|----------|----------|-----------|----------|-----------|
| Politics | 57% | 59% | 76% | 85% |
| Sports | 68% | 71% | 81% | 91% |
| Technology | 75% | 82% | 85% | 89% |
| Business | 73% | 78% | 80% | 87% |
| General | 74% | 79% | 81% | 88% |

**Key Finding**: Political headlines are most challenging (require historical context); sports headlines show largest improvement with fine-tuning (+23 points).

---

## Key Insights

### 1. Linguistic Sarcasm vs. Source Detection
Zero-shot models (74%) detect linguistic irony and exaggeration. Fine-tuned models (88%) additionally learn The Onion's characteristic writing style, vocabulary preferences, and topic distribution. This 14% gap explains why source-specific training is crucial.

### 2. Few-Shot Provides Best ROI
Just 5 examples improve accuracy by 5% over zero-shot—the highest return on investment per example provided. Ideal for scenarios requiring good accuracy without expensive fine-tuning infrastructure.

### 3. Complementary Strengths
- **Traditional ML**: Good with explicit keyword patterns; fails on novel words
- **Zero-Shot**: Strong semantic understanding; conservative on predictions
- **Few-Shot**: Balanced approach; adapts to task definition
- **Fine-Tuned**: Best overall; captures all pattern types

---

## Error Analysis

### Systematic Failure Mode 1: Meta-Satire
**Example**: "This 'brilliant' new technology will revolutionize everything"  
**Issue**: Real news written with sarcastic tone  
**Result**: All methods failed  
**Lesson**: Models learn source patterns more than linguistic markers

### Systematic Failure Mode 2: Context-Dependent Political Satire
**Example**: "Bush orders Iraq to disarm before start of war"  
**Issue**: Requires pre-2003 political knowledge  
**Result**: Zero-shot only method that failed  
**Lesson**: LLMs lack historical context without explicit training

### Systematic Failure Mode 3: Novel Vocabulary
**Example**: "New blockchain-based social platform disrupts industry"  
**Issue**: Unknown words outside training distribution  
**Result**: Traditional ML failed; neural methods succeeded  
**Lesson**: Feature-based approaches brittle to distributional shift

---

## Computational Requirements

| Method | Training Time | Inference Time | Hardware |
|--------|---|---|---|
| Traditional ML | <1 min | Instant | CPU |
| Zero-Shot LLM | N/A | ~500ms | Internet |
| Few-Shot LLM | N/A | ~600ms | Internet |
| Fine-Tuned | ~20 min | <50ms | GPU recommended |

---

## Limitations
1. **Binary Classification**: Only distinguishes satire vs. real; doesn't classify satire subtypes(satire, parody etc).
2. **English Only**: Single-language system; satire is culturally-specific.
3. **Temporal Sensitivity**: Political content requires historical context.
4. **U.S.-Centric**: American news sources only.

## Usage Examples

### Running Full Analysis
```bash
python sarcasm_project.py
```
Generates all visualizations, metrics, and trained models (~45 minutes).

### Using Web Interface
```bash
python interface.py
```
Interactive classification with real-time predictions.

### Inference with Fine-Tuned Model
```python
from transformers import pipeline

classifier = pipeline("text-classification", 
                     model="./sarcasm_model")
result = classifier("Your headline here")
```

---

## Dependencies

- anthropic >= 0.18.0
- transformers >= 4.30.0
- torch >= 2.0.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- gradio >= 4.0.0

