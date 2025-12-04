# 🔍 Semantic Leakage in Language Models  
### A Replication of *“Does Liking Yellow Imply Driving a School Bus?”* with an Indonesian Extension

This project is a **clean, end-to-end replication** of the core *semantic leakage* framework proposed in:

> **Does Liking Yellow Imply Driving a School Bus?  
> Semantic Leakage in Language Models** (2024)

with an added **cross-lingual extension to Indonesian** and an **interactive Streamlit UI** for exploration.

The goal of this project is to:
- Reproduce the **Leak-Rate** metric proposed in the paper
- Analyze **semantic leakage behavior in English**
- Extend the analysis to **Bahasa Indonesia**
- Provide a **fully interactive research demo interface**

---

## 🚀 What is *Semantic Leakage*?

Semantic leakage refers to the phenomenon where **language models unintentionally inject unrelated semantic concepts into their generation** simply because those concepts appear earlier in the prompt.

### Example

**Control Prompt**
> *“Complete the sentence: His favorite food is …”*

**Test Prompt**
> *“He likes koalas. His favorite food is …”*

Even though liking koalas has **no logical relation** to food, models often generate food related to animals (e.g., *eucalyptus*).  
This unintended influence is what we quantify as **semantic leakage**.

---

## 📏 Leak-Rate Metric

For each concept:
1. Generate multiple samples for **control prompts**
2. Generate multiple samples for **test prompts**
3. Measure similarity:
   - `sim(concept, control_generation)`
   - `sim(concept, test_generation)`
4. Define **Leak-Rate**:

| Case | Score |
|------|--------|
| `test > control` | 1 |
| `test < control` | 0 |
| equal | 0.5 |

Final Leak-Rate is the **average over all samples**, reported as a percentage.

---

## 🌍 What This Project Adds

Beyond basic replication, this project introduces:

✅ **Indonesian (Bahasa Indonesia) Extension**  
✅ **Side-by-side English vs Indonesian leakage behavior**  
✅ **Interactive Streamlit UI**  
✅ **Configurable temperature & sampling**  
✅ **Multiple embedding backends (SBERT / OpenAI)**  
✅ **Clean research logging & CSV export**

This makes the project both:
- 📚 **Publication-style**
- 💻 **Portfolio-ready & demo-friendly**

---

## 🗂 Project Structure

```text
semantic-leakage-replication/
├─ data/
│  ├─ prompts_en.csv          # English prompt pairs
│  └─ prompts_id.csv          # Indonesian prompt pairs
├─ results/                   # Auto-generated results
├─ config.py                  # API + experiment config
├─ semantic_leakage_core.py   # Core experiment logic
├─ run_experiments.py         # Batch runner
└─ app.py                     # Streamlit UI
