import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model (same as indexing)
model = SentenceTransformer(
    "perplexity-ai/pplx-embed-v1-0.6b",
    trust_remote_code=True
)

def load_index():
    index = faiss.read_index("faiss_index/index.bin")

    with open("faiss_index/data.pkl", "rb") as f:
        data = pickle.load(f)

    texts = data["texts"]
    metadatas = data["metadatas"]

    return index, texts, metadatas


def retrieve(query, top_k=5):
    # Load FAISS index, stored texts, and metadata
    index, texts, metadatas = load_index()

    # Embed query
    query_embedding = model.encode([query]).astype("float32")

    # Search FAISS index for top_k most similar vectors
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(texts[idx])

    return results


if __name__ == "__main__":
    query = "What is Artemis II mission?"

    results = retrieve(query)

    print("\nTop Results:\n")

    for i, r in enumerate(results):
        print(f"--- Result {i+1} ---\n")
        print(r[:500])
        print()