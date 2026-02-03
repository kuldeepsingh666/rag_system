"""
Evaluation Module for RAG System

This module implements comprehensive evaluation including
error categorization and hallucination stress testing.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


class ErrorCategory(Enum):
    """Categories of RAG errors."""
    RETRIEVAL_ERROR = "retrieval_error"
    CONTEXT_COMPRESSION_ERROR = "context_compression_error"
    LLM_REASONING_ERROR = "llm_reasoning_error"
    PROMPT_ENGINEERING_WEAKNESS = "prompt_engineering_weakness"
    NO_ERROR = "no_error"
    HALLUCINATION = "hallucination"


@dataclass
class EvaluationResult:
    """Result of evaluating a single question."""
    question: str
    question_type: str
    expected_answer: str
    generated_answer: str
    is_correct: bool
    retrieved_sources: List[str]
    expected_sources: List[str]
    retrieval_recall: float
    error_category: ErrorCategory
    error_analysis: str
    confidence: str
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "question_type": self.question_type,
            "expected_answer": self.expected_answer[:100] + "..." if len(self.expected_answer) > 100 else self.expected_answer,
            "generated_answer": self.generated_answer[:100] + "..." if len(self.generated_answer) > 100 else self.generated_answer,
            "is_correct": self.is_correct,
            "retrieval_recall": round(self.retrieval_recall, 2),
            "error_category": self.error_category.value,
            "error_analysis": self.error_analysis,
            "confidence": self.confidence
        }


@dataclass 
class TestCase:
    """A test case for evaluation."""
    question: str
    question_type: str  # "single-hop", "multi-hop", "ambiguous", "conflicting"
    expected_answer: str
    expected_sources: List[str]
    notes: str = ""


# Predefined test cases based on the corpus
TEST_CASES = [
    # Single-hop questions
    TestCase(
        question="When was TechVenture Inc. founded?",
        question_type="single-hop",
        expected_answer="March 15, 2018",
        expected_sources=["doc_01_techventure_overview.txt"],
        notes="Direct fact retrieval"
    ),
    TestCase(
        question="Who is the CEO of TechVenture Inc.?",
        question_type="single-hop",
        expected_answer="Sarah Chen",
        expected_sources=["doc_01_techventure_overview.txt", "doc_05_sarah_chen_bio.txt"],
        notes="Entity extraction"
    ),
    TestCase(
        question="What is the pricing for InsightEngine Starter Plan?",
        question_type="single-hop",
        expected_answer="$49/user/month (up to 10 users)",
        expected_sources=["doc_03_insightengine_launch.txt"],
        notes="Specific product pricing"
    ),
    
    # Multi-hop questions
    TestCase(
        question="Compare the 2022 and 2023 revenue for TechVenture and explain the growth.",
        question_type="multi-hop",
        expected_answer="2022: $45M, 2023: $78M, growth driven by InsightEngine and user expansion",
        expected_sources=["doc_01_techventure_overview.txt", "doc_04_q3_2023_financials.txt"],
        notes="Cross-document comparison"
    ),
    TestCase(
        question="What role did Robert Kim play at DataSync Solutions before the acquisition, and what is his current role at TechVenture?",
        question_type="multi-hop",
        expected_answer="Co-founder of DataSync, now VP of Data Products at TechVenture",
        expected_sources=["doc_02_datasync_acquisition.txt", "doc_18_robert_kim_bio.txt"],
        notes="Cross-document entity linking"
    ),
    TestCase(
        question="What was the original planned launch date for InsightEngine and what was the actual launch date?",
        question_type="multi-hop",
        expected_answer="Originally November 2022, delayed to January 10, 2023",
        expected_sources=["doc_16_launch_date_revision_memo.txt", "doc_03_insightengine_launch.txt"],
        notes="Requires reconciling dates from multiple docs"
    ),
    
    # Ambiguous/Conflicting questions
    TestCase(
        question="What is the best project management approach according to TechVenture customers?",
        question_type="ambiguous",
        expected_answer="Should indicate this is subjective or cite specific testimonials",
        expected_sources=["doc_11_case_study_globaltech.txt", "doc_17_case_study_retail_solutions.txt"],
        notes="Subjective question, should handle gracefully"
    ),
    TestCase(
        question="Who said 'The natural language query feature is game-changing' - was it from GlobalTech or Retail Solutions?",
        question_type="conflicting",
        expected_answer="Should note the quote appears attributed to different people/companies",
        expected_sources=["doc_11_case_study_globaltech.txt", "doc_17_case_study_retail_solutions.txt"],
        notes="Quote appears in both case studies with different attribution"
    ),
]

# Adversarial questions for hallucination testing
ADVERSARIAL_QUESTIONS = [
    {
        "question": "What was TechVenture's Q4 2025 revenue?",
        "type": "temporal_impossibility",
        "expected_behavior": "Insufficient evidence - future date beyond corpus",
        "guardrail_test": "Should detect date beyond document corpus"
    },
    {
        "question": "What did CEO John Smith say about TechVenture's Mars colonization project?",
        "type": "nonexistent_entity",
        "expected_behavior": "Insufficient evidence - CEO is Sarah Chen, no Mars project exists",
        "guardrail_test": "Should not fabricate quotes or entities"
    },
    {
        "question": "According to the TechVenture IPO prospectus, what is the company valuation?",
        "type": "fabricated_reference",
        "expected_behavior": "Insufficient evidence - no IPO prospectus in corpus",
        "guardrail_test": "Should verify document actually exists"
    },
    {
        "question": "What was the exact number of employees TechVenture had on June 15, 2023?",
        "type": "over_precision",
        "expected_behavior": "Should provide available data without inventing precise numbers",
        "guardrail_test": "Should not invent specific data points"
    },
    {
        "question": "What secret project is TechVenture working on that hasn't been announced?",
        "type": "speculation_inducing",
        "expected_behavior": "Insufficient evidence - should not speculate",
        "guardrail_test": "Should refuse to speculate about unannounced information"
    }
]


class RAGEvaluator:
    """
    Comprehensive evaluation with error categorization.
    """
    
    def __init__(self, rag_generator, llm: Optional[ChatOpenAI] = None):
        self.rag_generator = rag_generator
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
    
    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """
        Evaluate a single test case.
        
        Args:
            test_case: Test case to evaluate
            
        Returns:
            EvaluationResult
        """
        # Generate answer
        result = self.rag_generator.generate(test_case.question)
        
        # Calculate retrieval recall
        retrieved_sources = result.get("sources", [])
        expected_sources = test_case.expected_sources
        
        if expected_sources:
            matches = sum(1 for s in expected_sources if any(s in rs for rs in retrieved_sources))
            retrieval_recall = matches / len(expected_sources)
        else:
            retrieval_recall = 1.0 if retrieved_sources else 0.0
        
        # Determine correctness
        is_correct = self._check_correctness(
            test_case.expected_answer,
            result.get("answer", ""),
            test_case.question_type
        )
        
        # Categorize error
        error_category, error_analysis = self._categorize_error(
            test_case, result, retrieval_recall, is_correct
        )
        
        return EvaluationResult(
            question=test_case.question,
            question_type=test_case.question_type,
            expected_answer=test_case.expected_answer,
            generated_answer=result.get("answer", ""),
            is_correct=is_correct,
            retrieved_sources=retrieved_sources,
            expected_sources=expected_sources,
            retrieval_recall=retrieval_recall,
            error_category=error_category,
            error_analysis=error_analysis,
            confidence=result.get("confidence", "unknown")
        )
    
    def _check_correctness(
        self,
        expected: str,
        generated: str,
        question_type: str
    ) -> bool:
        """Check if generated answer is correct."""
        expected_lower = expected.lower()
        generated_lower = generated.lower()
        
        # For ambiguous/conflicting, check if it handles gracefully
        if question_type in ["ambiguous", "conflicting"]:
            hedging_phrases = [
                "insufficient", "unclear", "subjective", "depends",
                "both", "contradiction", "conflicting"
            ]
            return any(phrase in generated_lower for phrase in hedging_phrases)
        
        # For factual questions, check key facts
        # Extract key terms from expected answer
        key_terms = [t for t in expected_lower.split() if len(t) > 3]
        if not key_terms:
            return True
        
        matches = sum(1 for term in key_terms if term in generated_lower)
        return matches >= len(key_terms) * 0.5
    
    def _categorize_error(
        self,
        test_case: TestCase,
        result: Dict,
        retrieval_recall: float,
        is_correct: bool
    ) -> Tuple[ErrorCategory, str]:
        """Categorize the type of error if answer is incorrect."""
        if is_correct:
            return ErrorCategory.NO_ERROR, "Answer is correct"
        
        answer = result.get("answer", "").lower()
        
        # Check for hallucination
        if "insufficient" not in answer and retrieval_recall < 0.3:
            return ErrorCategory.HALLUCINATION, \
                "Answer provided without retrieving relevant sources"
        
        # Check retrieval error
        if retrieval_recall < 0.5:
            return ErrorCategory.RETRIEVAL_ERROR, \
                f"Only {retrieval_recall:.0%} of expected sources were retrieved"
        
        # Check if context was there but answer is wrong
        if retrieval_recall >= 0.5:
            context = result.get("context_preview", "")
            expected_terms = test_case.expected_answer.lower().split()[:5]
            terms_in_context = sum(1 for t in expected_terms if t in context.lower())
            
            if terms_in_context >= len(expected_terms) * 0.5:
                return ErrorCategory.LLM_REASONING_ERROR, \
                    "Relevant information was in context but LLM failed to synthesize correctly"
            else:
                return ErrorCategory.CONTEXT_COMPRESSION_ERROR, \
                    "Relevant info may have been lost during context compression"
        
        # Default to prompt engineering issue
        return ErrorCategory.PROMPT_ENGINEERING_WEAKNESS, \
            "Answer quality suggests prompt could be improved"
    
    def run_evaluation_suite(
        self,
        test_cases: List[TestCase] = None
    ) -> pd.DataFrame:
        """
        Run full evaluation suite.
        
        Args:
            test_cases: Optional custom test cases, uses defaults if None
            
        Returns:
            DataFrame with evaluation results
        """
        cases = test_cases or TEST_CASES
        results = []
        
        for tc in cases:
            print(f"Evaluating: {tc.question[:50]}...")
            eval_result = self.evaluate_single(tc)
            results.append(eval_result.to_dict())
        
        return pd.DataFrame(results)
    
    def run_hallucination_stress_test(self) -> pd.DataFrame:
        """
        Run adversarial hallucination tests.
        
        Returns:
            DataFrame with stress test results
        """
        results = []
        
        for adv in ADVERSARIAL_QUESTIONS:
            print(f"Stress testing: {adv['question'][:50]}...")
            
            result = self.rag_generator.generate(adv["question"])
            answer = result.get("answer", "").lower()
            
            # Check if guardrail triggered
            guardrail_success = (
                "insufficient" in answer or
                "cannot" in answer or
                "no information" in answer or
                "not found" in answer or
                "does not contain" in answer
            )
            
            results.append({
                "question": adv["question"],
                "adversarial_type": adv["type"],
                "expected_behavior": adv["expected_behavior"],
                "generated_answer": result.get("answer", "")[:200],
                "guardrail_triggered": guardrail_success,
                "guardrail_test": adv["guardrail_test"],
                "sources_retrieved": len(result.get("sources", [])),
                "confidence": result.get("confidence", "unknown")
            })
        
        return pd.DataFrame(results)
    
    def generate_evaluation_report(
        self,
        eval_results: pd.DataFrame,
        stress_results: pd.DataFrame
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            eval_results: Results from run_evaluation_suite
            stress_results: Results from run_hallucination_stress_test
            
        Returns:
            Formatted report string
        """
        report = "# RAG System Evaluation Report\n\n"
        
        # Summary statistics
        report += "## Summary\n\n"
        total = len(eval_results)
        correct = eval_results["is_correct"].sum()
        report += f"- **Accuracy**: {correct}/{total} ({correct/total*100:.1f}%)\n"
        report += f"- **Average Retrieval Recall**: {eval_results['retrieval_recall'].mean():.2f}\n\n"
        
        # Error breakdown
        report += "## Error Analysis\n\n"
        error_counts = eval_results["error_category"].value_counts()
        for error, count in error_counts.items():
            report += f"- {error}: {count}\n"
        
        # By question type
        report += "\n## Results by Question Type\n\n"
        by_type = eval_results.groupby("question_type")["is_correct"].agg(["sum", "count"])
        for qtype, row in by_type.iterrows():
            report += f"- **{qtype}**: {row['sum']}/{row['count']} correct\n"
        
        # Hallucination stress test
        report += "\n## Hallucination Stress Test\n\n"
        stress_pass = stress_results["guardrail_triggered"].sum()
        stress_total = len(stress_results)
        report += f"- **Guardrail Success Rate**: {stress_pass}/{stress_total} ({stress_pass/stress_total*100:.1f}%)\n\n"
        
        for _, row in stress_results.iterrows():
            status = "✅ PASS" if row["guardrail_triggered"] else "❌ FAIL"
            report += f"- [{status}] {row['adversarial_type']}: {row['question'][:50]}...\n"
        
        return report
