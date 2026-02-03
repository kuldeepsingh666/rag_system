# Configuration settings for RAG system
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding Models
EMBEDDING_MODELS = {
    "small": "text-embedding-3-small",
    "large": "text-embedding-3-large"
}
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# LLM Configuration
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 2000

# Chunking Configuration
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
CHUNKING_STRATEGIES = ["fixed", "semantic", "recursive", "sentence"]

# Retrieval Configuration
DEFAULT_TOP_K = 5
MMR_LAMBDA = 0.5

# FAISS Configuration
FAISS_INDEX_PATH = "faiss_index"

# Prompts
HALLUCINATION_GUARDRAIL_PROMPT = """You are a precise question-answering assistant. Your task is to answer questions 
ONLY using the provided context.

CRITICAL INSTRUCTIONS:
1. Base your answer EXCLUSIVELY on the provided context
2. If the context does not contain sufficient information, respond EXACTLY with:
   "Insufficient evidence. The provided documents do not contain information to answer this question."
3. Do NOT infer, speculate, or use external knowledge
4. Always cite the specific source document(s) supporting your answer using [Source: document_name] format
5. If sources contradict each other, acknowledge the contradiction explicitly

Context:
{context}

Question: {question}

Answer (with citations):"""

MULTI_QUERY_PROMPT = """You are an AI assistant that helps improve search queries.
Given the original question, generate 3 alternative versions of the question that might help retrieve relevant documents.
Each version should approach the question from a different angle.

Original question: {question}

Generate 3 alternative questions, one per line:"""

QUERY_DECOMPOSITION_PROMPT = """You are an AI assistant that breaks down complex questions into simpler sub-questions.
Given a complex question that may require multiple steps to answer, decompose it into simpler sub-questions.

Original question: {question}

Break this down into 2-4 simpler sub-questions that, when answered together, would answer the original question.
Return each sub-question on a new line:"""
