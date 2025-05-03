import faiss
import numpy as np
from typing import List, Tuple
import os
import pickle

class VectorSearch:
    def __init__(self, dimension: int):
        """
        Initialize the vector search index.
        
        Args:
            dimension: Dimension of the vectors to be indexed
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        # try HSNW -- doesn't respond? look into this
        # see this error: curl: (52) Empty reply from server
        # self.index = faiss.IndexHNSWFlat(dimension, 32)
        # self.index.hnsw.efSearch = 64
        # self.index.hnsw.efConstruction = 100
        self.texts = []
        self.metadata = []

    def add_vectors(self, vectors: np.ndarray, texts: List[str], metadata: List[dict]):
        """
        Add vectors and their associated texts and metadata to the index.
        
        Args:
            vectors: numpy array of shape (n_vectors, dimension)
            texts: List of text strings
            metadata: List of metadata dictionaries
        """
        if len(vectors) != len(texts) or len(vectors) != len(metadata):
            raise ValueError("Length of vectors, texts, and metadata must match")
            
        self.index.add(vectors)
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, dict, float]]:
        """
        Search for the k most similar vectors.
        
        Args:
            query_vector: numpy array of shape (dimension,)
            k: Number of results to return
            
        Returns:
            List of tuples (text, metadata, distance)
        """
        # Reshape query vector to 2D
        query_vector = query_vector.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 indicates no result
                results.append((
                    self.texts[idx],
                    self.metadata[idx],
                    float(distances[0][i])
                ))
        
        return results

    def save(self, directory: str):
        """
        Save the index and associated data to disk.
        
        Args:
            directory: Directory to save the files
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # Save texts and metadata
        with open(os.path.join(directory, "data.pkl"), "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "metadata": self.metadata
            }, f)

    @classmethod
    def load(cls, directory: str) -> 'VectorSearch':
        """
        Load a saved index and associated data from disk.
        
        Args:
            directory: Directory containing the saved files
            
        Returns:
            VectorSearch instance
        """
        # Load FAISS index
        index = faiss.read_index(os.path.join(directory, "index.faiss"))
        
        # Load texts and metadata
        with open(os.path.join(directory, "data.pkl"), "rb") as f:
            data = pickle.load(f)
        
        # Create instance
        instance = cls(index.d)
        instance.index = index
        instance.texts = data["texts"]
        instance.metadata = data["metadata"]
        
        return instance 