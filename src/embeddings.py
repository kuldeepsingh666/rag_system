"""
Embedding and Indexing Module for RAG System

This module handles embedding generation, model comparison,
and FAISS vector store operations.
"""

import os
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import pickle


@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models."""
    model_name: str
    dimensions: int
    description: str


class EmbeddingManager:
    """
    Manages embedding generation and model comparison.
    
    Supports multiple OpenAI embedding models for comparison.
    """
    
    MODELS = {
        "text-embedding-3-small": EmbeddingModelConfig(
            model_name="text-embedding-3-small",
            dimensions=1536,
            description="Faster, cheaper, good for most use cases"
        ),
        "text-embedding-3-large": EmbeddingModelConfig(
            model_name="text-embedding-3-large",
            dimensions=3072,
            description="Higher quality, best for semantic similarity"
        ),
        "text-embedding-ada-002": EmbeddingModelConfig(
            model_name="text-embedding-ada-002",
            dimensions=1536,
            description="Legacy model, good baseline"
        )
    }
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
        
        self.model_name = model_name
        self.config = self.MODELS[model_name]
        self.embeddings = OpenAIEmbeddings(model=model_name)
        self._embedding_cache = {}
    
    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Numpy array of embeddings
        """
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        return np.array(embeddings).astype('float32')
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.
        
        Args:
            query: Query string
            
        Returns:
            Numpy array of query embedding
        """
        # Check cache
        if query in self._embedding_cache:
            return self._embedding_cache[query]
        
        embedding = self.embeddings.embed_query(query)
        embedding_array = np.array(embedding).astype('float32')
        
        # Cache the result
        self._embedding_cache[query] = embedding_array
        return embedding_array
    
    @classmethod
    def compare_models(
        cls,
        documents: List[Document],
        test_queries: List[str],
        ground_truth: Optional[Dict[str, List[int]]] = None
    ) -> pd.DataFrame:
        """
        Empirical comparison of embedding models.
        
        Args:
            documents: Documents to embed
            test_queries: Queries to test
            ground_truth: Optional dict mapping query to relevant doc indices
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_name in ["text-embedding-3-small", "text-embedding-3-large"]:
            print(f"\nTesting {model_name}...")
            
            try:
                manager = cls(model_name)
                
                # Measure embedding time
                start_time = time.time()
                doc_embeddings = manager.embed_documents(documents)
                embed_time = time.time() - start_time
                
                # Build FAISS index
                index = faiss.IndexFlatIP(manager.config.dimensions)
                faiss.normalize_L2(doc_embeddings)
                index.add(doc_embeddings)
                
                # Test queries
                query_times = []
                for query in test_queries:
                    start = time.time()
                    query_embedding = manager.embed_query(query).reshape(1, -1)
                    faiss.normalize_L2(query_embedding)
                    _, _ = index.search(query_embedding, 5)
                    query_times.append(time.time() - start)
                
                results.append({
                    "model": model_name,
                    "dimensions": manager.config.dimensions,
                    "embed_time_s": round(embed_time, 3),
                    "avg_query_time_ms": round(np.mean(query_times) * 1000, 2),
                    "docs_per_second": round(len(documents) / embed_time, 1)
                })
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                results.append({
                    "model": model_name,
                    "error": str(e)
                })
        
        return pd.DataFrame(results)


class FAISSIndex:
    """
    FAISS vector store with multiple index types and persistence.
    """
    
    def __init__(self, embedding_dim: int = 1536, index_type: str = "flat"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = self._create_index()
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on type."""
        if self.index_type == "flat":
            # Exact search using inner product (cosine similarity after normalization)
            return faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "ivf":
            # Approximate search for larger corpora
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            # nlist = number of clusters (adjust based on corpus size)
            return faiss.IndexIVFFlat(quantizer, self.embedding_dim, 10, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def build_index(self, embeddings: np.ndarray, documents: List[Document]):
        """
        Build the FAISS index.
        
        Args:
            embeddings: Document embeddings
            documents: Document objects (for metadata)
        """
        self.documents = documents
        self.embeddings = embeddings.copy()
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf":
            if not self.index.is_trained:
                self.index.train(self.embeddings)
        
        # Add embeddings
        self.index.add(self.embeddings)
        print(f"Built index with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        # Normalize query
        query_normalized = query_embedding.copy().reshape(1, -1)
        faiss.normalize_L2(query_normalized)
        
        # Search
        scores, indices = self.index.search(query_normalized, k)
        
        # Return documents with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def search_with_mmr(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """
        Search with Maximal Marginal Relevance for diversity.
        
        Args:
            query_embedding: Query embedding
            k: Final number of results
            fetch_k: Initial candidates to fetch
            lambda_mult: Diversity parameter (0 = max diversity, 1 = max relevance)
            
        Returns:
            List of (Document, score) tuples
        """
        # Get initial candidates
        query_normalized = query_embedding.copy().reshape(1, -1)
        faiss.normalize_L2(query_normalized)
        scores, indices = self.index.search(query_normalized, fetch_k)
        
        if len(indices[0]) == 0:
            return []
        
        # MMR selection
        selected_indices = []
        selected_scores = []
        candidate_indices = list(indices[0])
        candidate_scores = list(scores[0])
        
        # Select first document (most relevant)
        if candidate_indices:
            best_idx = 0
            selected_indices.append(candidate_indices.pop(best_idx))
            selected_scores.append(candidate_scores.pop(best_idx))
        
        # Iteratively select remaining documents
        while len(selected_indices) < k and candidate_indices:
            best_score = float('-inf')
            best_idx = 0
            
            for i, (cand_idx, cand_score) in enumerate(zip(candidate_indices, candidate_scores)):
                if cand_idx < 0 or cand_idx >= len(self.embeddings):
                    continue
                    
                # Calculate MMR score
                relevance = cand_score
                
                # Calculate max similarity to already selected documents
                max_sim = 0
                for sel_idx in selected_indices:
                    if sel_idx >= 0 and sel_idx < len(self.embeddings):
                        sim = np.dot(self.embeddings[cand_idx], self.embeddings[sel_idx])
                        max_sim = max(max_sim, sim)
                
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected_indices.append(candidate_indices.pop(best_idx))
            selected_scores.append(candidate_scores.pop(best_idx))
        
        # Return documents
        results = []
        for idx, score in zip(selected_indices, selected_scores):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save(self, path: str):
        """Save index and documents to disk."""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        # Save documents and metadata
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "embeddings": self.embeddings,
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type
            }, f)
        
        print(f"Saved index to {path}")
    
    def load(self, path: str):
        """Load index and documents from disk."""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        # Load documents and metadata
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.embeddings = data["embeddings"]
            self.embedding_dim = data["embedding_dim"]
            self.index_type = data["index_type"]
        
        print(f"Loaded index with {len(self.documents)} documents")


def create_vector_store(
    chunks: List[Document],
    model_name: str = "text-embedding-3-small",
    index_type: str = "flat"
) -> Tuple[FAISSIndex, EmbeddingManager]:
    """
    Create a complete vector store from chunks.
    
    Args:
        chunks: List of document chunks
        model_name: Embedding model to use
        index_type: FAISS index type
        
    Returns:
        Tuple of (FAISSIndex, EmbeddingManager)
    """
    # Create embedding manager
    embedding_manager = EmbeddingManager(model_name)
    
    # Generate embeddings
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = embedding_manager.embed_documents(chunks)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Create and build index
    faiss_index = FAISSIndex(
        embedding_dim=embedding_manager.config.dimensions,
        index_type=index_type
    )
    faiss_index.build_index(embeddings, chunks)
    
    return faiss_index, embedding_manager
