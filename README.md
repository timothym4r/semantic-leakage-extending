# 🔍 Semantic Leakage in Language Models  
**Replication of “Does Liking Yellow Imply Driving a School Bus?” with an Indonesian Extension**

This project presents a **clean replication and cross-lingual extension** of the semantic leakage study introduced in:

> *Does Liking Yellow Imply Driving a School Bus? Semantic Leakage in Language Models* (Federici et al., 2024)

It measures how **irrelevant concepts in a prompt unintentionally influence model generations**, and extends the original study with:

- ✅ **Bahasa Indonesia evaluation**
- ✅ **Interactive Streamlit UI for visualization**

---

## 🧠 What is Semantic Leakage?

Semantic leakage occurs when a language model’s output is influenced by a concept that is **logically unrelated** to the task, simply because that concept appears earlier in the prompt.

**Example**

- Control prompt:  
  *“Complete the sentence: His favorite food is …”*

- Test prompt:  
  *“He likes koalas. His favorite food is …”*

If the model now generates food related to animals, the concept **“koalas” has leaked** into the output.

This project measures that effect **systematically and quantitatively**.

---

## 📏 Leak-Rate (Evaluation Metric)

For each concept:

1. Generate multiple completions for:
   - a **control prompt**
   - a **test prompt** (control + injected concept)
2. Compute semantic similarity between:
   - the **concept word**
   - each generated completion
3. Define Leak-Rate as the percentage of cases where:

```
similarity(concept, test) > similarity(concept, control)
```

- ~50% → no systematic leakage  
- Higher values → stronger semantic leakage  

---

## 🌍 Indonesian Extension

The exact same leakage evaluation pipeline is also applied to **Bahasa Indonesia**:

- Prompts are translated into Indonesian  
- The same concept structure is preserved  
- The same evaluation metric is used  

This enables **direct cross-lingual comparison of leakage behavior**.

---

## 📁 Project Structure

```
semantic-leakage-replication/
├─ data/
│  ├─ prompts_en.csv        # English prompt pairs
│  └─ prompts_id.csv        # Indonesian prompt pairs
├─ results/                 # Auto-generated leakage scores
├─ config.py                # Experiment configuration
├─ semantic_leakage_core.py # Generation, embeddings, Leak-Rate logic
├─ run_experiments.py       # Batch experiment runner
└─ app.py                   # Streamlit UI
```

---

## 📂 Dataset

The project uses paired **control** and **test** prompts stored in CSV files:

- `data/prompts_en.csv` — English prompt pairs  
- `data/prompts_id.csv` — Indonesian prompt pairs  

Each row follows this format:

```csv
id,concept,category,control_prompt,test_prompt
1,koalas,animals,"Complete the sentence: His favorite food is","Complete the sentence: He likes koalas. His favorite food is"
```

Indonesian example:

```csv
id,concept,category,control_prompt,test_prompt
1,koala,animals,"Lengkapi kalimat: Makanan favoritnya adalah","Lengkapi kalimat: Dia suka koala. Makanan favoritnya adalah"
```

The **only difference** between the control and test prompts is the **injected concept**, allowing precise isolation of semantic leakage.

---

## ⚙️ Running the Experiments

### ▶ Batch Evaluation

Runs the full leakage evaluation and saves results:

```bash
python run_experiments.py
```

This generates result files such as:

```
results/leakage_en_*.csv
results/leakage_id_*.csv
```

Each file contains per-concept Leak-Rates with metadata (language, category, embedding backend).

---

### 🖥 Interactive UI

Launch the dashboard with:

```bash
streamlit run app.py
```

The interface allows users to:

- Switch between **English and Indonesian**
- Compare **embedding backends**
- Adjust **temperature and sampling**
- Visualize:
  - Overall Leak-Rate
  - Category-level leakage patterns
  - Per-concept leakage rankings

This makes the project suitable for **presentations, demos, and qualitative inspection**.

---

## 🎯 What This Project Demonstrates

This project showcases:

- Behavioral evaluation of large language models  
- Embedding-based semantic similarity analysis  
- Multilingual robustness testing  
- Reproducible evaluation pipelines  
- Interactive research visualization with Streamlit  

It serves as a **foundation project in multilingual NLP, interpretability, and LLM behavior analysis**.

---

## 📜 Reference

```bibtex
@inproceedings{gonen-etal-2025-liking,
  title     = {Does Liking Yellow Imply Driving a School Bus? Semantic Leakage in Language Models},
  author    = {Gonen, Hila and Blevins, Terra and Liu, Alisa and Zettlemoyer, Luke and Smith, Noah A.},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year      = {2025},
  address   = {Albuquerque, New Mexico},
  publisher = {Association for Computational Linguistics},
}
```
