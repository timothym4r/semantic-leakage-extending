# config.py

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default models/settings – adjust as you like
OPENAI_CHAT_MODEL = "gpt-4.1-mini"   # or "gpt-4o" / "gpt-5.1-mini" etc
OPENAI_EMBED_MODEL = "text-embedding-3-large"

# Generation settings
NUM_SAMPLES = 5          # how many generations per prompt pair
TEMPERATURE = 1.0        # you can sweep later if you want

# Paths
DATA_DIR = "data"
RESULTS_DIR = "results"

# Random seed for reproducibility-ish
RANDOM_SEED = 42
