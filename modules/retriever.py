# retriever.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.documents = None
        self.embeddings = None

    def build_index(self, documents):
        self.documents = documents
        self.embeddings = self.embedder.encode(documents, convert_to_numpy=True).astype("float32")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def search(self, query, k=5):
        if self.index is None:
            raise Exception("Index not built. Call build_index() first.")

        q = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        dists, idxs = self.index.search(q, k)
        results = []
        for i in idxs[0]:
            results.append(self.documents[i])
        return results
