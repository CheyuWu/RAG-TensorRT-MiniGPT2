from sentence_transformers import SentenceTransformer
import faiss
from typing import List


class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.docs = []
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dim)

    def add_documents(self, documents: List[str]):
        embs = self.embedder.encode(documents, convert_to_numpy=True)
        self.index.add(embs.astype("float32"))
        self.docs.extend(documents)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if not self.docs:
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        # To avoid top_k exceeding document count
        k = min(top_k, len(self.docs))
        _, indices = self.index.search(q_emb, k)
        # Filter out invalid indices
        return [self.docs[i] for i in indices[0]]
