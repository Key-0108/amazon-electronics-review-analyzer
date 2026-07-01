import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Electronics Review Analyzer",
    page_icon="📊",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/amazon_reviews_with_sentiment.csv")
    neg = pd.read_csv("data/negative_reviews_with_themes.csv")
    themes = pd.read_csv("data/theme_counts.csv",
                         names=['theme', 'count'], header=0)
    return df, neg, themes

df, neg_df, theme_counts = load_data()

theme_names = {
    0: "Doesn't work as advertised",
    1: "Device & connectivity issues",
    2: "Durability & battery life",
    3: "Accessories fit & compatibility",
    4: "Defective on arrival",
    5: "Audio & sound quality"
}

# ── Header ────────────────────────────────────────────────────
st.title("📊 Amazon Electronics — AI Review Analyzer")
st.caption("NLP + HuggingFace Transformer + Claude API  |  Portfolio project by Keerthana")
st.divider()

# ── Row 1: KPI cards ──────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Reviews",    f"{len(df):,}")
col2.metric("Avg Rating",       f"{df['rating'].mean():.2f} / 5")
col3.metric("Negative Reviews", f"{round((df['hf_sentiment']=='negative').mean()*100,1)}%")
col4.metric("Positive Reviews", f"{round((df['hf_sentiment']=='positive').mean()*100,1)}%")
col5.metric("Complaint Themes", "6 identified")

st.divider()

# ── Row 2: Rating + Sentiment charts ──────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Rating distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    df['rating'].value_counts().sort_index().plot(
        kind='bar', ax=ax, color='#4A90D9', edgecolor='white', width=0.6)
    ax.set_xlabel("Stars")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

with col_b:
    st.subheader("Sentiment breakdown (HuggingFace model)")
    counts = df['hf_sentiment'].value_counts()
    colors = ['#27AE60' if l == 'positive' else '#E74C3C' if l == 'negative'
              else '#95A5A6' for l in counts.index]
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.pie(counts, labels=counts.index, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    plt.tight_layout()
    st.pyplot(fig2)

st.divider()

# ── Row 3: Model comparison ───────────────────────────────────
st.subheader("Model comparison — TextBlob vs HuggingFace Transformer")
col_c, col_d = st.columns([1, 2])

with col_c:
    comparison = pd.DataFrame({
        'Method':   ['TextBlob (baseline)', 'HuggingFace Transformer'],
        'Accuracy': ['67.4%', '80.4%'],
        'F1 Score': ['0.442', '0.613']
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.caption("Evaluated on 10,000 review sample")

with col_d:
    fig3, ax3 = plt.subplots(figsize=(6, 2.5))
    methods = ['TextBlob', 'HuggingFace\nTransformer']
    acc = [0.674, 0.804]
    f1  = [0.442, 0.613]
    x, w = [0, 1], 0.3
    b1 = ax3.bar([i-w/2 for i in x], acc, w, label='Accuracy',
                 color='#4A90D9', edgecolor='white')
    b2 = ax3.bar([i+w/2 for i in x], f1,  w, label='F1 (macro)',
                 color='#27AE60', edgecolor='white')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=9)
    for bar in b1 + b2:
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{bar.get_height():.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)

st.divider()

# ── Row 4: Complaint themes ───────────────────────────────────
st.subheader("Top complaint themes in negative reviews")
fig4, ax4 = plt.subplots(figsize=(9, 3.5))
theme_counts_sorted = theme_counts.sort_values('count', ascending=True)
ax4.barh(theme_counts_sorted['theme'], theme_counts_sorted['count'],
         color='#E74C3C', edgecolor='white')
ax4.set_xlabel("Number of negative reviews")
for i, v in enumerate(theme_counts_sorted['count']):
    ax4.text(v + 30, i, str(v), va='center', fontsize=9)
plt.tight_layout()
st.pyplot(fig4)

st.divider()

# ── Row 5: Claude brief ───────────────────────────────────────
st.subheader("AI-generated product health brief")

col_e, col_f = st.columns([1, 2])
with col_e:
    st.info("Click the button to generate a fresh executive brief using Claude API based on the analysis above.")
    generate = st.button("Generate brief with Claude", type="primary")

with col_f:
    if generate:
        with st.spinner("Claude is analyzing your data..."):
            try:
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                prompt = f"""You are a senior product analyst at an electronics marketplace.
Write a concise product health brief (max 250 words) for the VP of Electronics.

KEY METRICS:
- Total reviews: {len(df):,}
- Avg rating: {df['rating'].mean():.2f}/5
- Negative rate: {round((df['hf_sentiment']=='negative').mean()*100,1)}%

TOP COMPLAINT THEMES:
1. Device & connectivity issues — 6,056 (32%)
2. Defective on arrival — 3,232 (17%)
3. Audio & sound quality — 2,911 (15%)
4. Durability & battery life — 2,753 (14%)

FORMAT: Executive Summary | Top 3 Issues | 3 Recommendations | Risk if unaddressed"""

                msg = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(msg.content[0].text)
            except Exception as e:
                st.error(f"API error: {e}")
    else:
        # Show the pre-generated brief by default
        try:
    with open("memos/ai_product_brief.txt", "r") as f:
        st.write(f.read())
except:
    st.info("Click the button to generate the brief.")

st.divider()

# ── Row 6: Review explorer ────────────────────────────────────
st.subheader("Explore reviews")
col_g, col_h = st.columns(2)

with col_g:
    sentiment_filter = st.selectbox("Filter by sentiment", ["positive", "negative", "neutral"])
with col_h:
    n_reviews = st.slider("Number of reviews to show", 3, 20, 5)

filtered = df[df['hf_sentiment'] == sentiment_filter][['rating', 'title', 'text']].sample(
    min(n_reviews, len(df[df['hf_sentiment'] == sentiment_filter])),
    random_state=None
)
st.dataframe(filtered, use_container_width=True, height=300)