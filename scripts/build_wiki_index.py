from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    # Load Wikipedia
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

    # Chunk all articles
    print("Chunking articles...")
    all_chunks = []
    for article in wiki.select(range(50000)):  # start with 50k articles
        chunks = chunk_text(article["text"])
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")

    # Embed
    model = SentenceTransformer("BAAI/bge-m3")
    model.max_seq_length = 512
    embeddings = model.encode(
        all_chunks,
        batch_size=64,  # larger batch fine on GPU
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # Save
    os.makedirs("data/index", exist_ok=True)
    faiss.write_index(index, "data/index/wiki_bge.faiss")
    np.save("data/index/wiki_passages.npy", np.array(all_chunks))
    print("Done.")