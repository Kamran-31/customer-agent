"""
Zyvra FAISS Embedding Builder

Embedding model:
    sentence-transformers/all-MiniLM-L6-v2

Dimension:
    384

FAISS:
    IndexFlatIP

Because embeddings are normalized,
Inner Product is equivalent to cosine similarity.
"""

import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

EMBEDDINGS_DIR = BASE_DIR / "embeddings"

CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.json"

INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"

CONFIG_PATH = EMBEDDINGS_DIR / "embedding_config.json"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DIMENSION = 384


# ============================================================
# BUILD INDEX
# ============================================================

def main():

    print("=" * 60)
    print("ZYVRA FAISS EMBEDDING BUILDER")
    print("=" * 60)

    EMBEDDINGS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    if not CHUNKS_PATH.exists():

        raise FileNotFoundError(
            f"chunks.json not found:\n{CHUNKS_PATH}"
        )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    if not chunks:

        raise RuntimeError(
            "chunks.json contains no chunks."
        )

    print(
        f"Loaded {len(chunks)} knowledge chunks."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        f"Loading embedding model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    actual_dimension = (
        model.get_sentence_embedding_dimension()
    )

    if actual_dimension != DIMENSION:

        raise RuntimeError(
            f"Expected dimension {DIMENSION}, "
            f"received {actual_dimension}."
        )

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    texts = [
        item.get("text", "")
        for item in chunks
    ]

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print(
        "Generating 384-dimensional embeddings..."
    )

    embeddings = model.encode(

        texts,

        batch_size=32,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=True,
    ).astype("float32")

    print(
        f"Embedding matrix shape: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(
        DIMENSION
    )

    index.add(
        embeddings
    )

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {

        "embedding_model": MODEL_NAME,

        "dimension": DIMENSION,

        "index_type": "IndexFlatIP",

        "similarity": "cosine_similarity",

        "normalized_embeddings": True,

        "total_vectors": int(
            index.ntotal
        ),
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FAISS INDEX CREATED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Index: {INDEX_PATH}"
    )

    print(
        f"Vectors: {index.ntotal}"
    )

    print(
        f"Dimension: {index.d}"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
