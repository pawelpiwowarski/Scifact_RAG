import sys
import math
from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv

from evaluation.test import TestQuestion, load_tests

# Import rerank and stream_answer_question
from answer import stream_answer_question, fetch_context, rerank


load_dotenv(override=True)

MODEL = "gpt-5-nano"
db_name = "db"


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(
        description="Normalized Discounted Cumulative Gain (binary relevance)"
    )
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(description="Concise feedback on the answer quality")
    accuracy: float = Field(description="How factually correct is the answer? 1-5")
    completeness: float = Field(description="How complete is the answer? 1-5")
    relevance: float = Field(description="How relevant is the answer? 1-5")


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    keyword_lower = keyword.lower()
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0
        for doc in retrieved_docs[:k]
    ]
    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    test: TestQuestion, k: int = 10, use_reranker: bool = False
) -> RetrievalEval:
    """
    Evaluate retrieval performance, optionally using the reranker.
    """
    # 1. Fetch initial docs
    retrieved_docs = fetch_context(test.question)

    # 2. Rerank if requested
    if use_reranker:
        retrieved_docs = rerank(test.question, retrieved_docs)

    # Calculate metrics
    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    ndcg_scores = [
        calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords
    ]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (
        (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0
    )

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


def evaluate_answer(
    test: TestQuestion, use_reranker: bool = False
) -> tuple[AnswerEval, str, list]:
    """
    Evaluate answer quality using LLM-as-a-judge, optionally using the reranker.
    """
    full_answer = ""
    retrieved_docs = []

    # Pass use_reranker to stream_answer_question
    for chunk, docs in stream_answer_question(test.question, rerank_docs=use_reranker):
        full_answer += chunk
        if docs:
            retrieved_docs = docs

    generated_answer = full_answer

    # LLM judge prompt
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question: {test.question}\n\nGenerated Answer: {generated_answer}\n\nReference Answer: {test.reference_answer}\n\nEvaluate Accuracy, Completeness, and Relevance (1-5).""",
        },
    ]

    judge_response = completion(
        model=MODEL, messages=judge_messages, response_format=AnswerEval
    )

    answer_eval = AnswerEval.model_validate_json(
        judge_response.choices[0].message.content
    )

    return answer_eval, generated_answer, retrieved_docs


def evaluate_all_retrieval(use_reranker: bool = False):
    """Evaluate all retrieval tests."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test, use_reranker=use_reranker)
        progress = (index + 1) / total_tests
        yield test, result, progress


def evaluate_all_answers(use_reranker: bool = False):
    """Evaluate all answers to tests."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_answer(test, use_reranker=use_reranker)[0]
        progress = (index + 1) / total_tests
        yield test, result, progress
