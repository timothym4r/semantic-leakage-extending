# run_experiments.py

from semantic_leakage_core import run_leakage_experiment

def main():
    # Example: run English + Indonesian with SBERT embeddings
    for lang in ["en", "id"]:
        df = run_leakage_experiment(
            lang=lang,
            embedding_backend="sbert",
            num_samples=5,
            temperature=1.0,
        )
        print(df.head())

if __name__ == "__main__":
    main()
