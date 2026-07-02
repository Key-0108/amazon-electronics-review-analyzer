# Amazon Electronics - AI Review Analyzer

An end-to-end NLP and GenAI pipeline that analyzes 104,986 Amazon 
Electronics customer reviews to identify complaint themes and generate 
executive-level product health briefs using Claude API.

## Business Question
What are customers most unhappy about in Amazon Electronics products, 
and what actions should the product and operations teams take?

## Live Demo
[🚀 Open Streamlit App](https://amazon-electronics-review-analyzer-ja5vjafas6dwf9iazr5scb.streamlit.app/)

## Key Findings
- **18.2% negative review rate** across 104,986 reviews (avg rating 4.08/5)
- **Device & connectivity issues** is the #1 complaint - 6,056 reviews (32%)
- **Defective on arrival** affects 17% of complainers - a direct QC/logistics fix
- HuggingFace Transformer outperforms TextBlob by **13 points accuracy** 
  (80.4% vs 67.4%) - critical for catching sarcastic/mixed-language complaints

## What I Built
- **EDA** on 104,986 reviews (ratings, sentiment trends, verified vs unverified)
- **Sentiment classification** - compared 3 approaches:
  - Star-rating baseline
  - TextBlob (rule-based) - 67.4% accuracy, F1: 0.442
  - HuggingFace Transformer (cardiffnlp/twitter-roberta) - 80.4% accuracy, F1: 0.613
- **LDA topic modeling** - extracted 6 complaint themes from 19,070 negative reviews
- **Claude API** - generates PM-style executive product health brief from findings
- **Streamlit app** - interactive dashboard deployed publicly

## Tech Stack
| Layer | Tools |
|---|---|
| Data processing | Python, Pandas, NumPy |
| NLP | TextBlob, HuggingFace Transformers |
| Topic modeling | Scikit-learn (TF-IDF + LDA) |
| GenAI | Anthropic Claude API (claude-sonnet-4-6) |
| Visualization | Matplotlib, Seaborn |
| App | Streamlit |
| Database | PostgreSQL |
| BI Dashboard | Power BI |

## Project Structure
```
amazon-electronics-review-analyzer/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── amazon_reviews_electronics.csv
│   ├── amazon_reviews_with_sentiment.csv
│   ├── negative_reviews_with_themes.csv
│   ├── theme_counts.csv
│   ├── complaint_themes.png
│   └── model_comparison.png
├── images/
│   ├── dashboard_screenshot.png
│   ├── rating_distribution.png
│   ├── review_length_sentiment.png
│   ├── reviews_over_time.png
│   ├── textblob_confusion.png
│   └── verified_vs_unverified.png
├── memos/
│   └── ai_product_brief.txt
├── notebooks/
│   ├── 00_data_download.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_sentiment.ipynb
│   └── 03_theme_extraction.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset
Amazon Reviews 2023 — Electronics subset  
Source: McAuley-Lab/Amazon-Reviews-2023 (HuggingFace)  
Size: 104,986 reviews sampled from 22M+ record dataset