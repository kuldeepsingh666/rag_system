"""
Retrieval Module for RAG System

This module implements advanced retrieval strategies including
multi-query retrieval, hybrid search, and query decomposition.
"""
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import FAISSIndex, EmbeddingManager
from config.settings import MULTI_QUERY_PROMPT, QUERY_DECOMPOSITION_PROMPT


class QueryExpander:
    """
    Expands queries using LLM to improve retrieval coverage.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    def expand_query(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate query variations.
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations including original
        """
        prompt = MULTI_QUERY_PROMPT.format(question=query)
        
        try:
            response = self.llm.invoke(prompt)
            variations = [v.strip() for v in response.content.strip().split("\n") if v.strip()]
            return [query] + variations[:num_variations]
        except Exception as e:
            print(f"Query expansion error: {e}")
            return [query]
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose a complex query into sub-queries.
        
        Args:
            query: Complex query
            
        Returns:
            List of sub-queries
        """
        prompt = QUERY_DECOMPOSITION_PROMPT.format(question=query)
        
        try:
            response = self.llm.invoke(prompt)
            sub_queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
            # Remove numbering if present
            sub_queries = [q.lstrip("0123456789.) ") for q in sub_queries]
            return sub_queries if sub_queries else [query]
        except Exception as e:
            print(f"Query decomposition error: {e}")
            return [query]


class HybridRetriever:
    """
    Multi-stage retrieval pipeline with query expansion and MMR.
    """
    
    def __init__(
        self,
        faiss_index: FAISSIndex,
        embedding_manager: EmbeddingManager,
        llm: Optional[ChatOpenAI] = None
    ):
        self.faiss_index = faiss_index
        self.embedding_manager = embedding_manager
        self.query_expander = QueryExpander(llm)
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """
        Basic retrieval with optional MMR.
        
        Args:
            query: Query string
            k: Number of results
            use_mmr: Whether to use MMR for diversity
            mmr_lambda: MMR diversity parameter
            
        Returns:
            List of (Document, score) tuples
        """
        query_embedding = self.embedding_manager.embed_query(query)
        
        if use_mmr:
            return self.faiss_index.search_with_mmr(
                query_embedding, k=k, fetch_k=k*3, lambda_mult=mmr_lambda
            )
        else:
            return self.faiss_index.search(query_embedding, k=k)
    
    def multi_query_retrieve(
        self,
        query: str,
        k: int = 5,
        use_mmr: bool = True
    ) -> Tuple[List[Tuple[Document, float]], Dict]:
        """
        Retrieve using multiple query variations for better coverage.
        
        Args:
            query: Original query
            k: Number of final results
            use_mmr: Whether to use MMR
            
        Returns:
            Tuple of (results, metadata)
        """
        # Generate query variations
        query_variations = self.query_expander.expand_query(query)
        
        # Collect all results
        all_results = {}
        query_contributions = {q: [] for q in query_variations}
        
        for q in query_variations:
            results = self.retrieve(q, k=k*2, use_mmr=False)
            query_contributions[q] = [doc.metadata.get("source", "") for doc, _ in results[:3]]
            
            for doc, score in results:
                doc_id = id(doc)
                if doc_id not in all_results:
                    all_results[doc_id] = (doc, score, [q])
                else:
                    existing = all_results[doc_id]
                    # Boost score for documents found by multiple queries
                    new_score = max(existing[1], score) + 0.1
                    all_results[doc_id] = (doc, new_score, existing[2] + [q])
        
        # Sort by score
        sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
        
        # Apply MMR on final results if requested
        if use_mmr:
            final_results = self._apply_mmr_reranking(
                [(doc, score) for doc, score, _ in sorted_results],
                k=k
            )
        else:
            final_results = [(doc, score) for doc, score, _ in sorted_results[:k]]
        
        metadata = {
            "query_variations": query_variations,
            "query_contributions": query_contributions,
            "total_unique_docs": len(all_results)
        }
        
        return final_results, metadata
    
    def _apply_mmr_reranking(
        self,
        results: List[Tuple[Document, float]],
        k: int,
        lambda_mult: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """Apply MMR reranking to results."""
        if len(results) <= k:
            return results
        
        selected = []
        remaining = list(results)
        
        # Select first document
        if remaining:
            best = max(remaining, key=lambda x: x[1])
            selected.append(best)
            remaining.remove(best)
        
        # Iteratively select remaining
        while len(selected) < k and remaining:
            best_score = float('-inf')
            best_doc = None
            
            for doc, score in remaining:
                relevance = score
                
                # Calculate diversity (simple text overlap penalty)
                max_overlap = 0
                for sel_doc, _ in selected:
                    overlap = len(set(doc.page_content.lower().split()) & 
                                set(sel_doc.page_content.lower().split()))
                    overlap_ratio = overlap / max(len(doc.page_content.split()), 1)
                    max_overlap = max(max_overlap, overlap_ratio)
                
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_overlap
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = (doc, score)
            
            if best_doc:
                selected.append(best_doc)
                remaining.remove(best_doc)
        
        return selected
    
    def retrieve_for_multihop(
        self,
        query: str,
        k: int = 5
    ) -> Tuple[List[Tuple[Document, float]], Dict]:
        """
        Retrieve documents for multi-hop questions.
        
        Decomposes the query and retrieves for each sub-query.
        
        Args:
            query: Complex query
            k: Results per sub-query
            
        Returns:
            Tuple of (combined results, metadata)
        """
        # Decompose query
        sub_queries = self.query_expander.decompose_query(query)
        
        # Retrieve for each sub-query
        all_results = {}
        sub_query_results = {}
        
        for sub_q in sub_queries:
            results = self.retrieve(sub_q, k=k, use_mmr=True)
            sub_query_results[sub_q] = results
            
            for doc, score in results:
                doc_id = id(doc)
                if doc_id not in all_results:
                    all_results[doc_id] = (doc, score, [sub_q])
                else:
                    existing = all_results[doc_id]
                    new_score = existing[1] + score * 0.5
                    all_results[doc_id] = (doc, new_score, existing[2] + [sub_q])
        
        # Sort and deduplicate
        sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
        final_results = [(doc, score) for doc, score, _ in sorted_results[:k*2]]
        
        metadata = {
            "original_query": query,
            "sub_queries": sub_queries,
            "sub_query_results": {
                sq: [(doc.metadata.get("source", ""), score) 
                     for doc, score in results[:3]]
                for sq, results in sub_query_results.items()
            },
            "total_unique_docs": len(all_results)
        }
        
        return final_results, metadata


class ContextFormatter:
    """
    Formats retrieved documents into context for LLM.
    """
    
    @staticmethod
    def format_with_sources(
        results: List[Tuple[Document, float]],
        include_scores: bool = False,
        max_chars: int = 8000
    ) -> str:
        """
        Format results with source attribution.
        
        Args:
            results: Retrieved documents with scores
            include_scores: Whether to include similarity scores
            max_chars: Maximum characters in context
            
        Returns:
            Formatted context string
        """
        context_parts = []
        total_chars = 0
        
        for i, (doc, score) in enumerate(results):
            source = doc.metadata.get("filename", doc.metadata.get("source", f"Document {i+1}"))
            title = doc.metadata.get("title", "")[:50]
            
            header = f"[Source {i+1}: {source}]"
            if title:
                header += f" - {title}"
            if include_scores:
                header += f" (relevance: {score:.3f})"
            
            content = doc.page_content
            
            # Check if adding this would exceed limit
            entry = f"{header}\n{content}\n"
            if total_chars + len(entry) > max_chars:
                # Truncate content
                remaining = max_chars - total_chars - len(header) - 50
                if remaining > 100:
                    content = content[:remaining] + "..."
                    entry = f"{header}\n{content}\n"
                else:
                    break
            
            context_parts.append(entry)
            total_chars += len(entry)
        
        return "\n---\n".join(context_parts)
    
    @staticmethod
    def format_for_multihop(
        results: List[Tuple[Document, float]],
        sub_queries: List[str],
        sub_query_mapping: Dict
    ) -> str:
        """
        Format context for multi-hop questions with sub-query organization.
        
        Args:
            results: All retrieved documents
            sub_queries: List of sub-queries
            sub_query_mapping: Mapping of sub-query to relevant doc sources
            
        Returns:
            Formatted context with sub-query grouping
        """
        context = "# Retrieved Context for Multi-hop Question\n\n"
        
        for i, sub_q in enumerate(sub_queries):
            context += f"## Sub-question {i+1}: {sub_q}\n\n"
            
            # Find relevant documents for this sub-query
            relevant_sources = sub_query_mapping.get(sub_q, [])
            for doc, score in results:
                source = doc.metadata.get("source", doc.metadata.get("filename", ""))
                if source in [s for s, _ in relevant_sources]:
                    context += f"[{source}]\n{doc.page_content[:500]}...\n\n"
            
            context += "---\n\n"
        
        return context
