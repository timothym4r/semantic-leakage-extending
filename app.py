# app.py

import os

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from semantic_leakage_core import run_leakage_experiment
from config import RESULTS_DIR

st.set_page_config(
    page_title="Semantic Leakage Explorer",
    layout="wide",
)

st.title("🔍 Semantic Leakage in Language Models")
st.markdown(
    """
Replicating the core setup of  
**“Does Liking Yellow Imply Driving a School Bus? Semantic Leakage in Language Models”**  
with extensions to **Indonesian**.
"""
)

# Sidebar controls
st.sidebar.header("Experiment Settings")

lang = st.sidebar.selectbox("Language", ["en", "id"], format_func=lambda x: "English" if x == "en" else "Indonesian")
embedding_backend = st.sidebar.selectbox("Embedding backend", ["sbert", "openai"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, step=0.1, value=1.0)
num_samples = st.sidebar.slider("Samples per prompt pair", min_value=1, max_value=10, value=5)

run_button = st.sidebar.button("Run Experiment")

st.sidebar.markdown("---")
st.sidebar.markdown("Results will be cached to the `results/` folder.")

# Helper to load cached results if present
def load_cached_results(lang, embedding_backend, temperature):
    fname = f"leakage_{lang}_{embedding_backend}_temp{temperature}.csv".replace(".", "-")
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


if run_button:
    with st.spinner("Running leakage experiment..."):
        df_leak = run_leakage_experiment(
            lang=lang,
            embedding_backend=embedding_backend,
            num_samples=num_samples,
            temperature=temperature,
        )
else:
    df_leak = load_cached_results(lang, embedding_backend, temperature)

if df_leak is None or df_leak.empty:
    st.info("Run an experiment from the sidebar to see results.")
else:
    st.subheader("Overall Leak-Rate")

    overall = df_leak["leak_rate"].mean()
    st.metric(
        label=f"Average Leak-Rate ({'English' if lang=='en' else 'Indonesian'})",
        value=f"{overall:.1f}%",
        help="Leak-Rate > 50% indicates systematic semantic leakage."
    )

    # Per-category breakdown
    st.subheader("Leak-Rate by Category")
    cat_df = df_leak.groupby("category")["leak_rate"].mean().reset_index()

    fig, ax = plt.subplots()
    ax.bar(cat_df["category"], cat_df["leak_rate"])
    ax.set_ylabel("Leak-Rate (%)")
    ax.set_xlabel("Category")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(cat_df["category"], rotation=45, ha="right")
    st.pyplot(fig)

    # Per-concept table
    st.subheader("Per-Concept Leak-Rate")
    st.dataframe(
        df_leak.sort_values("leak_rate", ascending=False).reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown(
        """
**Interpretation:**  
- Values near **50%** → no systematic leakage.  
- Values **well above 50%** → concept strongly leaks into the generation.
"""
    )

    st.markdown("---")
    st.caption("You can inspect individual generations by logging them inside `run_leakage_experiment` or by saving a separate generations CSV.")
