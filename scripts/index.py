from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datasets import load_from_disk

if __name__ == "__main__":
    model = SentenceTransformer("BAAI/bge-m3")
    model.max_seq_length = 512

    english = load_from_disk("data/tydiqa_english_train")

    print(english)

    # Extract passages to index
    passages = english["context"]  # or whatever the text field is called

    # Encode — BGE expects a prompt prefix for retrieval
    corpus_embeddings = model.encode(
        passages,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True  # needed for cosine similarity via inner product
    )

    # Build FAISS index
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine since embeddings are normalized
    index.add(corpus_embeddings.astype(np.float32))

    # Save index + passages to disk
    faiss.write_index(index, "data/index/english_bge.faiss")
    np.save("data/index/english_passages.npy", np.array(passages))  # keep passages aligned with index