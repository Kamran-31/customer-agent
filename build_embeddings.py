"""Build a real FAISS index for the Zyvra knowledge base.

This script uses:
  sentence-transformers/all-MiniLM-L6-v2
  384-dimensional normalized embeddings
  FAISS IndexFlatIP (inner product = cosine similarity after normalization)

Run in an environment where sentence-transformers and faiss-cpu are installed.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "chunks.json"
INDEX_PATH = BASE_DIR / "faiss.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSION = 384


def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    model = SentenceTransformer(MODEL_NAME)
    assert model.get_sentence_embedding_dimension() == DIMENSION

    texts = [item["text"] for item in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(DIMENSION)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    print(f"Created {INDEX_PATH}")
    print(f"Vectors: {index.ntotal}")
    print(f"Dimension: {index.d}")
    print(f"Model: {MODEL_NAME}")


if __name__ == "__main__":
    main()
