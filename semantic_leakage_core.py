# semantic_leakage_core.py

import os
import random
from dataclasses import dataclass
from typing import List, Literal, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBED_MODEL,
    DATA_DIR,
    RESULTS_DIR,
    NUM_SAMPLES,
    TEMPERATURE,
    RANDOM_SEED,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --------------------
# Data structures
# --------------------

@dataclass
class PromptInstance:
    """One concept with paired control/test prompts."""
    id: int
    concept: str
    category: str
    control_prompt: str
    test_prompt: str


def load_prompt_dataset(lang: Literal["en", "id"] = "en") -> List[PromptInstance]:
    fname = f"prompts_{lang}.csv"
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)

    required_cols = {"id", "concept", "category", "control_prompt", "test_prompt"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns in {path}: {missing}")

    instances = []
    for _, row in df.iterrows():
        instances.append(
            PromptInstance(
                id=int(row["id"]),
                concept=str(row["concept"]),
                category=str(row["category"]),
                control_prompt=str(row["control_prompt"]),
                test_prompt=str(row["test_prompt"]),
            )
        )
    return instances


# --------------------
# OpenAI helpers
# --------------------

def get_openai_client() -> OpenAI:
    if OPENAI_API_KEY is None:
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env")
    return OpenAI(api_key=OPENAI_API_KEY)


def generate_text(
    client: OpenAI,
    prompt: str,
    n: int = 1,
    temperature: float = TEMPERATURE,
    model: str = OPENAI_CHAT_MODEL,
) -> List[str]:
    """
    Simple chat-completion call. Uses one user message with the prompt.
    """
    responses = []
    for _ in range(n):
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=64,  # sentence completion
        )
        text = completion.choices[0].message.content.strip()
        responses.append(postprocess_generation(prompt, text))
    return responses


def postprocess_generation(prompt: str, generation: str) -> str:
    """
    Mirror paper preprocessing:
    - remove repeated prompt if the model echoed it
    - truncate at first period
    """
    # Remove prompt echo (rough heuristic)
    if generation.startswith(prompt):
        generation = generation[len(prompt):].lstrip()

    # Truncate at first period
    if "." in generation:
        generation = generation.split(".", 1)[0]

    return generation.strip()


# --------------------
# Embeddings & similarity
# --------------------

class EmbeddingBackend:
    def __init__(self, kind: Literal["sbert", "openai"] = "sbert"):
        self.kind = kind
        self.client = None
        self.sbert_model = None

        if kind == "sbert":
            # Paper uses SentenceBERT via HF; we pick a generic English model.
            # You can swap to `efederici/sentence-bert-base` if you like.
            self.sbert_model = SentenceTransformer("all-mpnet-base-v2")
        elif kind == "openai":
            self.client = get_openai_client()
        else:
            raise ValueError(f"Unknown embedding backend: {kind}")

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.kind == "sbert":
            embs = self.sbert_model.encode(texts, show_progress_bar=False)
            return np.array(embs)
        elif self.kind == "openai":
            resp = self.client.embeddings.create(
                model=OPENAI_EMBED_MODEL,
                input=texts,
            )
            embs = [d.embedding for d in resp.data]
            return np.array(embs)
        else:
            raise ValueError(f"Unknown embedding backend: {self.kind}")


def pairwise_concept_similarity(
    embedder: EmbeddingBackend,
    concept: str,
    control_gens: List[str],
    test_gens: List[str],
) -> List[Tuple[float, float]]:
    """
    Returns [(sim_control, sim_test), ...] for each matched pair of generations.
    """
    if len(control_gens) != len(test_gens):
        raise ValueError("control_gens and test_gens must be same length")

    pairs = list(zip(control_gens, test_gens))
    texts = [concept] + [g for pair in pairs for g in pair]  # concept + all generations
    embs = embedder.embed(texts)

    concept_emb = embs[0:1]
    gen_embs = embs[1:]

    sims = cosine_similarity(concept_emb, gen_embs)[0]  # shape (2 * n,)
    result = []
    for i in range(len(pairs)):
        sim_control = sims[2 * i]
        sim_test = sims[2 * i + 1]
        result.append((sim_control, sim_test))

    return result


# --------------------
# Leak-Rate computation
# --------------------

def leak_rate_from_sim_pairs(sim_pairs: List[Tuple[float, float]]) -> float:
    """
    Implements Leak-Rate as in Eq. (1) of the paper:
        1 if sim_test > sim_control
        0 if sim_test < sim_control
        0.5 if equal
    Returns percentage in [0, 100].
    """
    scores = []
    for sim_control, sim_test in sim_pairs:
        if sim_test > sim_control:
            scores.append(1.0)
        elif sim_test < sim_control:
            scores.append(0.0)
        else:
            scores.append(0.5)
    if not scores:
        return 50.0
    return 100.0 * float(np.mean(scores))


def run_leakage_experiment(
    lang: Literal["en", "id"] = "en",
    embedding_backend: Literal["sbert", "openai"] = "sbert",
    num_samples: int = NUM_SAMPLES,
    temperature: float = TEMPERATURE,
) -> pd.DataFrame:
    """
    Runs leakage experiment over all prompt instances for a given language.
    Returns a DataFrame with per-instance and per-category results.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading prompts for language: {lang}")
    instances = load_prompt_dataset(lang=lang)

    client = get_openai_client()
    embedder = EmbeddingBackend(kind=embedding_backend)

    rows = []

    for inst in tqdm(instances, desc=f"Running leakage ({lang}, {embedding_backend})"):
        # Generate control & test outputs
        control_gens = generate_text(
            client, inst.control_prompt, n=num_samples, temperature=temperature
        )
        test_gens = generate_text(
            client, inst.test_prompt, n=num_samples, temperature=temperature
        )

        # Compute similarities vs concept
        sim_pairs = pairwise_concept_similarity(
            embedder, inst.concept, control_gens, test_gens
        )

        leak = leak_rate_from_sim_pairs(sim_pairs)

        for i, (gen_c, gen_t) in enumerate(zip(control_gens, test_gens)):
            sim_c, sim_t = sim_pairs[i]
            rows.append(
                {
                    "id": inst.id,
                    "concept": inst.concept,
                    "category": inst.category,
                    "language": lang,
                    "embedding_backend": embedding_backend,
                    "temperature": temperature,
                    "sample_idx": i,
                    "control_prompt": inst.control_prompt,
                    "test_prompt": inst.test_prompt,
                    "control_generation": gen_c,
                    "test_generation": gen_t,
                    "sim_control": sim_c,
                    "sim_test": sim_t,
                }
            )

        # Add one aggregate row for this instance (duplicated concept-level leak)
        rows.append(
            {
                "id": inst.id,
                "concept": inst.concept,
                "category": inst.category,
                "language": lang,
                "embedding_backend": embedding_backend,
                "temperature": temperature,
                "sample_idx": "ALL",
                "control_prompt": inst.control_prompt,
                "test_prompt": inst.test_prompt,
                "control_generation": "",
                "test_generation": "",
                "sim_control": np.nan,
                "sim_test": np.nan,
            }
        )

    df = pd.DataFrame(rows)

    # Compute per-concept leak-rate
    leak_results = []
    for (cid, concept), group in df[df["sample_idx"] != "ALL"].groupby(["id", "concept"]):
        sim_pairs = list(zip(group["sim_control"], group["sim_test"]))
        leak = leak_rate_from_sim_pairs(sim_pairs)
        category = group["category"].iloc[0]
        leak_results.append(
            {
                "id": cid,
                "concept": concept,
                "category": category,
                "language": lang,
                "embedding_backend": embedding_backend,
                "temperature": temperature,
                "leak_rate": leak,
            }
        )

    df_leak = pd.DataFrame(leak_results)

    # Save
    out_csv = os.path.join(
        RESULTS_DIR,
        f"leakage_{lang}_{embedding_backend}_temp{temperature}.csv".replace(".", "-"),
    )
    df_leak.to_csv(out_csv, index=False)
    print(f"Saved per-concept leak results to {out_csv}")

    return df_leak
