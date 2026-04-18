from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def chunk_text(text, chunk_size=200, overlap=50):
    """Identical to build_wiki_index.py."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


if __name__ == "__main__":
    # Load Telugu Wikipedia
    print("Loading Telugu Wikipedia...")
    telugu_wiki = load_dataset("wikimedia/wikipedia", "20231101.te", split="train")
    print(f"Total Telugu Wikipedia articles: {len(telugu_wiki)}")

    # Chunk all articles
    NUM_ARTICLES = 20000  # cap for Kaggle/Colab; increase on SCC
    print(f"Chunking {NUM_ARTICLES} articles...")
    all_chunks = []
    for article in telugu_wiki.select(range(min(NUM_ARTICLES, len(telugu_wiki)))):
        chunks = chunk_text(article["text"])
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")

    # Embed
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    print(f"Using device: {device}")
    model.max_seq_length = 512
    embeddings = model.encode(
        all_chunks,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # Save
    os.makedirs("data/index", exist_ok=True)
    faiss.write_index(index, "data/index/telugu_wiki_bge.faiss")
    np.save("data/index/telugu_wiki_passages.npy", np.array(all_chunks))
    print("Done.")
