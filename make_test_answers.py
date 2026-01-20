import argparse
import json
from typing import List, Literal
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from pydantic import Field
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate RAG evaluation questions")
    parser.add_argument("--CORPUS_FILE", type=str, default="./scifact/corpus.jsonl")
    parser.add_argument("--OUTPUT_FILE", type=str, default="./evaluation/tests.jsonl")
    parser.add_argument("--MAX_TOKENS", type=int, default=128_000)
    parser.add_argument("--TOTAL_NUMBER_OF_QUESTIONS", type=int, default=100)
    parser.add_argument("--QUESTIONS_PER_BATCH", type=int, default=25)
    parser.add_argument("--MODEL", type=str, default="gpt-5-nano")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------


class TestQuestion(BaseModel):
    """A test question data point for RAG evaluation."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: List[str] = Field(
        description="Unique keywords from the text that should appear in the answer"
    )
    reference_answer: str = Field(
        description="A concise reference answer based strictly on the text"
    )
    category: Literal[
        "direct_fact", "temporal", "comparison", "spanning", "relationship"
    ] = Field(description="The type of reasoning required to answer the question")
    # CRITICAL: We need this to calculate MRR/NDCG later
    source_doc_ids: List[str] = Field(
        description="The exact '_id' of the document(s) used to generate this question"
    )


class TestDataset(BaseModel):
    questions: List[TestQuestion]


# ---------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------
def num_tokens_from_string(string: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))


# ---------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------
def process_batch(llm, batch_docs, questions_per_batch: int):
    context_text = ""
    for doc in batch_docs:
        context_text += (
            f"<document id='{doc['_id']}'>\n"
            f"{doc.get('title', '')}\n"
            f"{doc.get('text', '')}\n"
            f"</document>\n\n"
        )

    system_prompt = f"""
You are an expert at creating datasets for Information Retrieval (RAG) evaluation.

You will be given {len(batch_docs)} documents.

Generate EXACTLY {questions_per_batch} questions and return them as valid JSON
matching this schema:

{TestDataset.model_json_schema()}

Rules:
1. `source_doc_ids` MUST exactly match the <document id="..."> values.
2. Use a mix of categories.
3. `reference_answer` must be self-contained and strictly grounded in the text.
4. If category is "comparison" or "spanning", source_doc_ids MUST contain more than one document ID.
"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_text),
        ]
    )

    return response.questions


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    load_dotenv(override=True)

    CORPUS_FILE = args.CORPUS_FILE
    OUTPUT_FILE = args.OUTPUT_FILE
    MAX_TOKENS = args.MAX_TOKENS
    TOTAL_NUMBER_OF_QUESTIONS = args.TOTAL_NUMBER_OF_QUESTIONS
    QUESTIONS_PER_BATCH = args.QUESTIONS_PER_BATCH
    MODEL = args.MODEL

    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
    ).with_structured_output(TestDataset)

    current_batch_docs = []
    current_tokens = 0
    total_questions_generated = 0

    print(f"Starting generation using {MODEL}")
    print(f"Target questions: {TOTAL_NUMBER_OF_QUESTIONS}")

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if total_questions_generated >= TOTAL_NUMBER_OF_QUESTIONS:
                break

            if not line.strip():
                continue

            doc = json.loads(line)

            doc_str = (
                f"<document id='{doc['_id']}'>\n"
                f"{doc.get('title', '')}\n"
                f"{doc.get('text', '')}\n"
                f"</document>\n\n"
            )
            doc_tokens = num_tokens_from_string(doc_str, MODEL)

            if current_tokens + doc_tokens > MAX_TOKENS and current_batch_docs:
                questions = process_batch(llm, current_batch_docs, QUESTIONS_PER_BATCH)

                with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
                    for q in questions:
                        if total_questions_generated >= TOTAL_NUMBER_OF_QUESTIONS:
                            break
                        out.write(q.model_dump_json() + "\n")
                        total_questions_generated += 1

                print(f"Saved batch → total questions: {total_questions_generated}")

                current_batch_docs = []
                current_tokens = 0

            current_batch_docs.append(doc)
            current_tokens += doc_tokens

    # Final batch
    if current_batch_docs and total_questions_generated < TOTAL_NUMBER_OF_QUESTIONS:
        questions = process_batch(llm, current_batch_docs, QUESTIONS_PER_BATCH)

        with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
            for q in questions:
                if total_questions_generated >= TOTAL_NUMBER_OF_QUESTIONS:
                    break
                out.write(q.model_dump_json() + "\n")
                total_questions_generated += 1

    print(f"Done! Generated {total_questions_generated} questions → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
