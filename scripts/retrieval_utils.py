import faiss
import numpy as np


def load_index(index_path, passages_path):
    """Load a FAISS index and its aligned passages from disk.
    Identical to the load_index function in rag.py.
    Used by both rag.py (English index) and rag_monolingual.py (Telugu index).
    """
    index = faiss.read_index(index_path)
    passages = np.load(passages_path, allow_pickle=True).tolist()
    return index, passages


def retrieve(query, retriever, index, passages, k=3):
    """Retrieve the top-k passages most similar to the query.
    Identical to the retrieve function in rag.py.
    Used by both rag.py (English index) and rag_monolingual.py (Telugu index).
    """
    query_embedding = retriever.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)
    scores, indices = index.search(query_embedding, k)
    return [passages[i] for i in indices[0]]
