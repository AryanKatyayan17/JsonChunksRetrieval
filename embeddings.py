import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from load_chunks import load_all_chunks


# Load embedding model
model = SentenceTransformer(
    "perplexity-ai/pplx-embed-v1-0.6b",
    trust_remote_code=True
)

def build_faiss_index(texts):
    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index

    # Get the dimensionality of each embedding vector
    # embeddings.shape = (num_documents, embedding_size)
    # We extract embedding_size (e.g., 768)
    dimension = embeddings.shape[1]

    # Create a FAISS index using L2 (Euclidean distance)
    # IndexFlatL2 = brute-force search (compares query with ALL vectors)
    # 'dimension' tells FAISS the size of each vector
    index = faiss.IndexFlatL2(dimension)

    # Add all embedding vectors to the FAISS index
    index.add(embeddings)

    return index, embeddings


def save_index(index, texts, metadatas):
    os.makedirs("faiss_index", exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, "faiss_index/index.bin")

    # Save texts + metadata
    with open("faiss_index/data.pkl", "wb") as f:
        pickle.dump({
            "texts": texts,
            "metadatas": metadatas
        }, f)


if __name__ == "__main__":
    texts, metadatas = load_all_chunks("json data")

    index, embeddings = build_faiss_index(texts)

    save_index(index, texts, metadatas)

    print("FAISS index created with", len(texts), "documents")