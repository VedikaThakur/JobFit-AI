# core/rag_pipeline.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Union

class Embedder:
    def __init__(self):
        # Fully offline, free, unlimited model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dim = 384

    def embed(self, text: str) -> np.ndarray:
        """Embed single text. Returns numpy array."""
        embedding = self.model.encode(
            text, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        return np.array(embedding)  # Ensure it's always numpy array


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: List[str] = []

    def chunk_text(self, text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def build_index(self, embeddings: Union[List, np.ndarray], chunks: List[str]):
        self.chunks = chunks
        
        # Convert to numpy array safely
        embeddings_array = np.array(embeddings)
        
        d = self.embedder.dim
        self.index = faiss.IndexFlatL2(d)
        
        # FAISS requires float32
        self.index.add(embeddings_array.astype(np.float32))  # type: ignore

    def retrieve(self, query_emb: np.ndarray, k: int = 6) -> List[str]:
        if self.index is None or self.index.ntotal == 0:
            return self.chunks[:k] if self.chunks else []
        
        # Ensure query is 2D array
        query_array = np.array([query_emb]).astype(np.float32)
        
        distances, indices = self.index.search(query_array, k)  # type: ignore
        
        # Return valid chunks
        result = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                result.append(self.chunks[int(idx)])
        return result
