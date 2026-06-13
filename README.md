# 🧠 Sentiment Analysis: Judge Emotion About Brands and Products

An end-to-end NLP classification pipeline that automatically detects the sentiment of tweets directed at brands and products, built following the **CRISP-DM** methodology.

---

## 📌 Project Overview

Social media generates millions of brand-related opinions daily — manually tracking public sentiment at scale is infeasible. This project builds a machine learning classifier trained on the [CrowdFlower dataset](https://www.crowdflower.com/data-for-everyone/) (9,093 tweets, August 2013) to classify tweet sentiment into four categories:

| Label | Class |
|-------|-------|
| 0 | Negative emotion |
| 1 | No emotion toward brand or product |
| 2 | Positive emotion |
| 3 | I can't tell |

---

## ❓ Business Questions

1. **BQ1** — Can we automatically detect whether a tweet expresses emotion toward a brand?
2. **BQ2** — Which sentiment class is the hardest for the model to predict correctly?
3. **BQ3** — Between Naive Bayes and Logistic Regression, which model is better suited for brand sentiment classification?

---

## 📁 Repository Structure

```
├── Sentiment_Analysis_Restructured.ipynb   # Main notebook
├── judge-1377884607_tweet_product_company.csv  # Raw dataset
├── Sentiment_Analysis_CRISP_DM.pptx        # Presentation deck
└── README.md
```

---

## 🔄 CRISP-DM Pipeline

```
Business Understanding → Data Understanding → EDA → Data Preparation → Modelling → Evaluation → Conclusions
```

### 1. Business Understanding
Defined the problem, business objective, and three core business questions.

### 2. Data Understanding
- Loaded and inspected the dataset (shape, dtypes, missing values)
- Renamed columns for clarity: `Tweet`, `Target`, `Sentiment`
- Explored class and target distributions

### 3. Exploratory Data Analysis
- **Univariate:** Sentiment distribution — dataset is heavily imbalanced (59% No Emotion)
- **Bivariate:** Sentiment by brand/product — Apple's ecosystem dominates emotional tweets
- **Multivariate:** Cross-tabulation of Sentiment × Target

### 4. Data Preprocessing
| Step | Detail |
|------|--------|
| Text Cleaning | Lowercase, remove numbers & punctuation, strip whitespace |
| Lemmatization | NLTK `WordNetLemmatizer` |
| Stopword Removal | NLTK English stopwords |
| Label Encoding | 4-class integer mapping |
| Train-Test Split | 80/20 stratified split |
| TF-IDF Vectorization | 5,000 features, unigrams + bigrams |

### 5. Modelling
Three classifiers trained in order of complexity:
- **Multinomial Naive Bayes** — probabilistic baseline
- **Logistic Regression (default)** — `C=1.0`, `solver=lbfgs`
- **Logistic Regression (tuned)** — `GridSearchCV` over `C`, `solver`, `class_weight`; scored on F1-Macro

### 6. Model Evaluation
- Confusion matrices for all three models
- Comparison across Accuracy, F1-Macro, F1-Weighted
- Per-class F1 heatmap to identify hardest classes
- Direct answers to all three business questions

---

## 📊 Key Results

| Model | Accuracy | F1 Macro | F1 Weighted |
|-------|----------|----------|-------------|
| Naive Bayes | 0.6636 | 0.3330 | 0.6180 |
| LR Default | 0.6839 | 0.3760 | 0.6511 |
| **LR Tuned** | **0.6416** | **0.4407** | **0.6505** |

> **Primary metric: F1-Macro** — accuracy is misleading due to class imbalance.

### Business Question Answers
- **BQ1:** ✅ Yes — ~39% of tweets carry emotion. Automated detection is feasible at scale.
- **BQ2:** ⚠️ *Negative* and *Can't Tell* are the hardest classes (lowest F1 across all models) due to severe underrepresentation.
- **BQ3:** 🏆 Tuned Logistic Regression wins — highest F1-Macro and most balanced class-level performance.

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

---

## 🚀 Usage

1. Place `judge-1377884607_tweet_product_company.csv` in the same directory as the notebook.
2. Run all cells top to bottom — the pipeline is fully sequential.
3. Charts are saved as `.png` files in the working directory for use in reporting.

---

## 💡 Recommendations

- **Deploy Tuned LR** as the production sentiment classifier.
- **Address class imbalance** with SMOTE or aggressive `class_weight='balanced'` for Negative/Can't Tell.
- **Monitor negative sentiment** closely — low volume but high reputational risk.
- **Expand features** — add tweet metadata (hashtags, emoji, follower count).
- Consider **brand-specific models** given Apple's dominance in this dataset.

---

## 🔭 Next Steps

| Priority | Action |
|----------|--------|
| 🔴 High | Apply SMOTE to Negative & Can't Tell classes |
| 🔴 High | Experiment with BERT / RoBERTa transformers |
| 🟡 Medium | Add URL, emoji, and hashtag features |
| 🟡 Medium | Build real-time inference API (FastAPI) |
| 🟢 Low | Collect more recent tweet data (post-2013) |
| 🟢 Low | Train brand-specific classifiers |

---

## 👤 Author

**Joel Muoki Andrew**  
Data Scientist | NLP & Machine Learning
# Sentiment-Analysis--NLP
