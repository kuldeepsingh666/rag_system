"""
Data Preparation Module for RAG System

This module handles document loading, preprocessing, and dynamic chunking
with multiple strategies for optimal retrieval performance.
"""

import os
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader
)

from langchain_core.documents import Document

import re


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    strategy: Literal["fixed", "semantic", "recursive", "sentence"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: Optional[List[str]] = None
    

class DynamicChunker:
    """
    Implements multiple chunking strategies that adapt to document structure.
    
    Strategies:
    - fixed: Simple fixed-size chunks
    - semantic: Splits on paragraph/section boundaries
    - recursive: Recursive character splitting with overlap
    - sentence: Sentence-based chunking for QA tasks
    """
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._init_splitters()
    
    def _init_splitters(self):
        """Initialize text splitters for each strategy."""
        self.splitters = {
            "fixed": CharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separator="\n"
            ),
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=self.config.separators or ["\n\n", "\n", ". ", " ", ""]
            ),
            "semantic": RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
            )
        }
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Dynamically select chunking strategy based on document type and content.
        
        Args:
            document: LangChain Document object
            
        Returns:
            List of chunked Document objects with metadata
        """
        # Detect document type and select strategy
        strategy = self._detect_optimal_strategy(document)
        splitter = self.splitters.get(strategy, self.splitters["recursive"])
        
        # Split document
        chunks = splitter.split_documents([document])
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunking_strategy": strategy,
                "chunk_size": len(chunk.page_content),
                "source_file": document.metadata.get("source", "unknown")
            })
        
        return chunks
    
    def _detect_optimal_strategy(self, document: Document) -> str:
        """
        Detect the optimal chunking strategy based on document structure.
        
        Args:
            document: Document to analyze
            
        Returns:
            Strategy name
        """
        content = document.page_content
        
        # Check for markdown headers
        if re.search(r'^#+\s', content, re.MULTILINE):
            return "semantic"
        
        # Check for structured content (lists, tables)
        if re.search(r'^\s*[-*•]\s', content, re.MULTILINE):
            return "semantic"
        
        # Check for short paragraphs (email/memo style)
        paragraphs = content.split("\n\n")
        avg_para_len = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
        if avg_para_len < 200:
            return "semantic"
        
        # Default to recursive for general content
        return "recursive"
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of Documents
            
        Returns:
            List of all chunks
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
    
    def compare_strategies(self, document: Document) -> Dict[str, Dict]:
        """
        Compare different chunking strategies on a document.
        
        Args:
            document: Document to chunk
            
        Returns:
            Dictionary with statistics for each strategy
        """
        results = {}
        
        for strategy_name, splitter in self.splitters.items():
            chunks = splitter.split_documents([document])
            chunk_sizes = [len(c.page_content) for c in chunks]
            
            results[strategy_name] = {
                "num_chunks": len(chunks),
                "avg_chunk_size": sum(chunk_sizes) / max(len(chunk_sizes), 1),
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                "total_chars": sum(chunk_sizes)
            }
        
        return results


class DocumentLoader:
    """
    Loads documents from various sources with metadata extraction.
    """
    
    SUPPORTED_EXTENSIONS = {
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".pdf": PyPDFLoader
    }
    
    def __init__(self, documents_dir: str):
        self.documents_dir = documents_dir
    
    def load_all(self) -> List[Document]:
        """
        Load all supported documents from the directory.
        
        Returns:
            List of Document objects with metadata
        """
        documents = []
        
        for filename in os.listdir(self.documents_dir):
            filepath = os.path.join(self.documents_dir, filename)
            
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[1].lower()
                
                if ext in self.SUPPORTED_EXTENSIONS:
                    try:
                        loader_class = self.SUPPORTED_EXTENSIONS[ext]
                        loader = loader_class(filepath)
                        docs = loader.load()
                        
                        # Extract and add metadata
                        for doc in docs:
                            doc.metadata.update(self._extract_metadata(doc, filename))
                            documents.append(doc)
                            
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
        
        return documents
    
    def _extract_metadata(self, document: Document, filename: str) -> Dict:
        """
        Extract metadata from document content.
        
        Args:
            document: Document object
            filename: Source filename
            
        Returns:
            Metadata dictionary
        """
        content = document.page_content
        metadata = {
            "filename": filename,
            "char_count": len(content),
            "word_count": len(content.split()),
            "line_count": content.count("\n") + 1
        }
        
        # Try to extract title (first non-empty line)
        lines = content.strip().split("\n")
        if lines:
            metadata["title"] = lines[0].strip()[:100]
        
        # Try to extract date mentions
        date_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        dates = re.findall(date_pattern, content)
        if dates:
            metadata["mentioned_dates"] = dates[:5]  # First 5 dates
        
        # Extract entities mentioned
        # Simple pattern for company/person names (capitalized words)
        entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', content))
        metadata["entities"] = list(entities)[:10]
        
        return metadata


def prepare_documents(documents_dir: str, chunking_config: ChunkingConfig = None) -> List[Document]:
    """
    Full document preparation pipeline.
    
    Args:
        documents_dir: Path to documents directory
        chunking_config: Optional chunking configuration
        
    Returns:
        List of processed and chunked documents
    """
    # Load documents
    loader = DocumentLoader(documents_dir)
    documents = loader.load_all()
    print(f"Loaded {len(documents)} documents")
    
    # Chunk documents
    chunker = DynamicChunker(chunking_config)
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    return chunks


class ChunkingComparison:
    """
    Utility class for comparing chunking strategies empirically.
    """
    
    @staticmethod
    def compare_on_corpus(documents: List[Document], configs: List[ChunkingConfig]) -> Dict:
        """
        Compare multiple chunking configurations on a corpus.
        
        Args:
            documents: List of documents
            configs: List of configurations to compare
            
        Returns:
            Comparison results
        """
        results = {}
        
        for config in configs:
            chunker = DynamicChunker(config)
            all_chunks = chunker.chunk_documents(documents)
            
            chunk_sizes = [len(c.page_content) for c in all_chunks]
            
            results[f"{config.strategy}_{config.chunk_size}"] = {
                "total_chunks": len(all_chunks),
                "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                "std_chunk_size": (
                    (sum((s - sum(chunk_sizes)/len(chunk_sizes))**2 for s in chunk_sizes) / len(chunk_sizes))**0.5
                    if chunk_sizes else 0
                ),
                "config": {
                    "strategy": config.strategy,
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap
                }
            }
        
        return results
