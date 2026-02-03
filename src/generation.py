"""
Generation Module for RAG System

This module implements context engineering, hallucination prevention,
and LangGraph-based RAG pipeline with multi-hop support.
"""

from typing import List, Dict, Optional, Tuple, TypedDict, Annotated
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
import operator
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import HALLUCINATION_GUARDRAIL_PROMPT
from src.retrieval import HybridRetriever, ContextFormatter


class ContextEngineer:
    """
    Prepares optimal context for LLM generation.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
    
    def compress_context(
        self,
        results: List[Tuple[Document, float]],
        query: str,
        max_chars: int = 4000
    ) -> str:
        """
        Compress and extract most relevant passages.
        
        Args:
            results: Retrieved documents with scores
            query: Original query
            max_chars: Maximum characters in output
            
        Returns:
            Compressed context string
        """
        # Simple compression: take most relevant chunks up to limit
        context_parts = []
        total_chars = 0
        
        for doc, score in results:
            if total_chars >= max_chars:
                break
            
            content = doc.page_content
            source = doc.metadata.get("filename", "Unknown")
            
            # Truncate if needed
            remaining = max_chars - total_chars
            if len(content) > remaining:
                content = content[:remaining] + "..."
            
            context_parts.append(f"[Source: {source}]\n{content}")
            total_chars += len(content) + len(source) + 20
        
        return "\n\n---\n\n".join(context_parts)
    
    def extract_relevant_sentences(
        self,
        results: List[Tuple[Document, float]],
        query: str,
        max_sentences: int = 10
    ) -> str:
        """
        Extract the most relevant sentences from retrieved documents.
        
        Args:
            results: Retrieved documents
            query: Query for relevance scoring
            max_sentences: Maximum sentences to extract
            
        Returns:
            Extracted sentences with sources
        """
        prompt = f"""Given the query and document content, extract the {max_sentences} most relevant sentences.
        
Query: {query}

Documents:
"""
        for i, (doc, score) in enumerate(results[:5]):
            source = doc.metadata.get("filename", f"Doc{i}")
            prompt += f"\n[{source}]\n{doc.page_content[:1000]}\n"
        
        prompt += f"\n\nExtract the {max_sentences} most relevant sentences, preserving source attribution:"
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            # Fallback to simple compression
            return self.compress_context(results, query)


# LangGraph State for RAG Pipeline
class RAGState(TypedDict):
    """State for the RAG pipeline."""
    query: str
    query_type: str  # "single" or "multihop"
    sub_queries: List[str]
    retrieved_docs: List[Tuple[Document, float]]
    context: str
    answer: str
    sources: List[str]
    confidence: str
    reasoning_steps: List[str]
    error: Optional[str]


class RAGGenerator:
    """
    LangGraph-based RAG generator with hallucination prevention.
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: Optional[ChatOpenAI] = None
    ):
        self.retriever = retriever
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.context_engineer = ContextEngineer(self.llm)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("retrieve_single", self._retrieve_single)
        workflow.add_node("retrieve_multihop", self._retrieve_multihop)
        workflow.add_node("compress_context", self._compress_context)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("verify_answer", self._verify_answer)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Add edges
        workflow.add_conditional_edges(
            "analyze_query",
            self._route_by_query_type,
            {
                "single": "retrieve_single",
                "multihop": "retrieve_multihop"
            }
        )
        workflow.add_edge("retrieve_single", "compress_context")
        workflow.add_edge("retrieve_multihop", "compress_context")
        workflow.add_edge("compress_context", "generate_answer")
        workflow.add_edge("generate_answer", "verify_answer")
        workflow.add_edge("verify_answer", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: RAGState) -> RAGState:
        """Analyze query to determine if single or multi-hop."""
        query = state["query"]
        
        # Simple heuristics for multi-hop detection
        multihop_indicators = [
            " and ", " then ", " after ", " before ",
            "compare", "comparison", "both", "between",
            "relationship", "how does", "what caused",
            "multiple", "connection", "linked"
        ]
        
        is_multihop = any(ind in query.lower() for ind in multihop_indicators)
        
        state["query_type"] = "multihop" if is_multihop else "single"
        state["reasoning_steps"] = [f"Query type: {state['query_type']}"]
        
        return state
    
    def _route_by_query_type(self, state: RAGState) -> str:
        """Route based on query type."""
        return state["query_type"]
    
    def _retrieve_single(self, state: RAGState) -> RAGState:
        """Retrieve for single-hop query."""
        results, metadata = self.retriever.multi_query_retrieve(
            state["query"], k=5, use_mmr=True
        )
        
        state["retrieved_docs"] = results
        state["reasoning_steps"].append(
            f"Retrieved {len(results)} documents using multi-query expansion"
        )
        
        return state
    
    def _retrieve_multihop(self, state: RAGState) -> RAGState:
        """Retrieve for multi-hop query with decomposition."""
        results, metadata = self.retriever.retrieve_for_multihop(
            state["query"], k=5
        )
        
        state["retrieved_docs"] = results
        state["sub_queries"] = metadata.get("sub_queries", [])
        state["reasoning_steps"].append(
            f"Decomposed into sub-queries: {state['sub_queries']}"
        )
        state["reasoning_steps"].append(
            f"Retrieved {len(results)} documents across sub-queries"
        )
        
        return state
    
    def _compress_context(self, state: RAGState) -> RAGState:
        """Compress retrieved context."""
        context = ContextFormatter.format_with_sources(
            state["retrieved_docs"],
            include_scores=False,
            max_chars=6000
        )
        
        state["context"] = context
        state["sources"] = [
            doc.metadata.get("filename", "Unknown") 
            for doc, _ in state["retrieved_docs"]
        ]
        
        return state
    
    def _generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer with hallucination guardrails."""
        prompt = HALLUCINATION_GUARDRAIL_PROMPT.format(
            context=state["context"],
            question=state["query"]
        )
        
        try:
            response = self.llm.invoke(prompt)
            state["answer"] = response.content
            state["reasoning_steps"].append("Generated answer with guardrail prompt")
        except Exception as e:
            state["answer"] = f"Error generating answer: {e}"
            state["error"] = str(e)
        
        return state
    
    def _verify_answer(self, state: RAGState) -> RAGState:
        """Verify answer against context for consistency."""
        answer = state["answer"]
        
        # Check for insufficient evidence response
        if "insufficient evidence" in answer.lower():
            state["confidence"] = "low"
            state["reasoning_steps"].append(
                "Guardrail triggered: insufficient evidence in context"
            )
        else:
            # Simple verification: check if answer mentions sources
            source_mentions = sum(1 for s in state["sources"] if s.lower() in answer.lower())
            if source_mentions > 0:
                state["confidence"] = "high"
                state["reasoning_steps"].append(
                    f"Answer cites {source_mentions} source(s)"
                )
            else:
                state["confidence"] = "medium"
                state["reasoning_steps"].append(
                    "Answer generated but source citations may be implicit"
                )
        
        return state
    
    def generate(self, query: str) -> Dict:
        """
        Generate an answer for the query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with answer, sources, confidence, and reasoning
        """
        initial_state: RAGState = {
            "query": query,
            "query_type": "",
            "sub_queries": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "sources": [],
            "confidence": "",
            "reasoning_steps": [],
            "error": None
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "query": final_state["query"],
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "confidence": final_state["confidence"],
            "query_type": final_state["query_type"],
            "sub_queries": final_state["sub_queries"],
            "reasoning_steps": final_state["reasoning_steps"],
            "context_preview": final_state["context"][:500] + "..." if len(final_state["context"]) > 500 else final_state["context"],
            "error": final_state["error"]
        }


class MultiHopGenerator:
    """
    Specialized generator for multi-hop questions with explicit reasoning.
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: Optional[ChatOpenAI] = None
    ):
        self.retriever = retriever
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
    
    def answer_multihop(self, query: str) -> Dict:
        """
        Answer a multi-hop question with explicit reasoning trace.
        
        Args:
            query: Multi-hop question
            
        Returns:
            Answer with reasoning trace
        """
        # Step 1: Decompose query
        decompose_prompt = f"""Break down this complex question into simpler sub-questions:

Question: {query}

List 2-4 sub-questions that need to be answered to fully answer the main question:"""
        
        decompose_response = self.llm.invoke(decompose_prompt)
        sub_questions = [
            q.strip().lstrip("0123456789.) -")
            for q in decompose_response.content.strip().split("\n")
            if q.strip()
        ]
        
        # Step 2: Answer each sub-question
        intermediate_answers = []
        all_sources = []
        
        for sub_q in sub_questions:
            results, _ = self.retriever.multi_query_retrieve(sub_q, k=3)
            context = ContextFormatter.format_with_sources(results, max_chars=2000)
            all_sources.extend([doc.metadata.get("filename", "") for doc, _ in results])
            
            sub_prompt = HALLUCINATION_GUARDRAIL_PROMPT.format(
                context=context,
                question=sub_q
            )
            
            sub_response = self.llm.invoke(sub_prompt)
            intermediate_answers.append({
                "question": sub_q,
                "answer": sub_response.content,
                "sources": [doc.metadata.get("filename", "") for doc, _ in results]
            })
        
        # Step 3: Synthesize final answer
        synthesis_prompt = f"""Based on the following intermediate answers, provide a comprehensive answer to the original question.

Original Question: {query}

Intermediate Answers:
"""
        for i, ia in enumerate(intermediate_answers):
            synthesis_prompt += f"\n{i+1}. {ia['question']}\nAnswer: {ia['answer']}\n"
        
        synthesis_prompt += """
Synthesize a final answer that:
1. Addresses the original question completely
2. Cites the sources appropriately
3. Notes any contradictions or gaps in the information
4. States "Insufficient evidence" if the intermediate answers don't support a conclusion

Final Answer:"""
        
        final_response = self.llm.invoke(synthesis_prompt)
        
        return {
            "query": query,
            "sub_questions": sub_questions,
            "intermediate_answers": intermediate_answers,
            "final_answer": final_response.content,
            "all_sources": list(set(all_sources)),
            "reasoning_trace": {
                "decomposition": sub_questions,
                "intermediate_steps": [
                    {"q": ia["question"], "a": ia["answer"][:200] + "..."}
                    for ia in intermediate_answers
                ]
            }
        }
    
    def handle_contradictions(
        self,
        query: str,
        results: List[Tuple[Document, float]]
    ) -> Dict:
        """
        Explicitly handle contradictory information in sources.
        
        Args:
            query: User query
            results: Retrieved documents
            
        Returns:
            Answer with contradiction analysis
        """
        context = ContextFormatter.format_with_sources(results)
        
        prompt = f"""Analyze the following sources for contradictions and answer the question.

Context:
{context}

Question: {query}

Instructions:
1. First, identify if there are any contradictions between sources
2. If contradictions exist, explicitly list them
3. Determine which source is more authoritative or recent
4. Provide an answer based on the most reliable information
5. If contradictions cannot be resolved, state "Insufficient evidence due to contradictory sources"

Analysis and Answer:"""
        
        response = self.llm.invoke(prompt)
        
        return {
            "query": query,
            "answer": response.content,
            "sources": [doc.metadata.get("filename", "") for doc, _ in results],
            "contradiction_analysis": True
        }
