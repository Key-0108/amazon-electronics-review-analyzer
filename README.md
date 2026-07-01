# Amazon Electronics — AI Review Analyzer

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
- **Device & connectivity issues** is the #1 complaint — 6,056 reviews (32%)
- **Defective on arrival** affects 17% of complainers — a direct QC/logistics fix
- HuggingFace Transformer outperforms TextBlob by **13 points accuracy** 
  (80.4% vs 67.4%) — critical for catching sarcastic/mixed-language complaints

## What I Built
- **EDA** on 104,986 reviews (ratings, sentiment trends, verified vs unverified)
- **Sentiment classification** — compared 3 approaches:
  - Star-rating baseline
  - TextBlob (rule-based) — 67.4% accuracy, F1: 0.442
  - HuggingFace Transformer (cardiffnlp/twitter-roberta) — 80.4% accuracy, F1: 0.613
- **LDA topic modeling** — extracted 6 complaint themes from 19,070 negative reviews
- **Claude API** — generates PM-style executive product health brief from findings
- **Streamlit app** — interactive dashboard deployed publicly

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
amazon-reviews-analysis/
├── data/                          # processed CSVs
├── notebooks/
|   ├── 00_data_download           
│   ├── 01_eda.ipynb               # exploratory data analysis
│   ├── 02_sentiment_nlp.ipynb     # sentiment modeling & comparison
│   └── 03_theme_extraction.ipynb  # LDA complaint theme extraction
├── app/
│   └── streamlit_app.py           # deployed interactive dashboard
├── memos/
│   └── ai_product_brief.txt       # Claude-generated executive brief
├── requirements.txt
└── README.md

**=== PROJECT 1A SUMMARY ===**
Dataset: 104,986 reviews
Avg rating: 4.08 / 5
Negative reviews: 19,070 (18.2%)
TextBlob accuracy: 67.4% | F1: 0.442
HuggingFace accuracy: 80.4% | F1: 0.613

Top complaint theme: Device & connectivity issues (32%)
Most actionable: Defective on arrival (17%) — direct QC fix

Files saved:
  data/amazon_reviews_with_sentiment.csv
  data/negative_reviews_with_themes.csv
  data/theme_counts.csv
  data/complaint_themes.png
  data/model_comparison.png
  memos/ai_product_brief.txt