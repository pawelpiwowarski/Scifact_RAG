import json
import argparse
import sys
from typing import List
from dotenv import load_dotenv

# Import both providers
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest a JSONL corpus into Chroma")

    parser.add_argument(
        "--corpus_file_path", type=str, default="./scifact/corpus.jsonl"
    )
    parser.add_argument(
        "--embedding_provider",
        type=str,
        default="openai",
        choices=["openai", "huggingface"],
        help="The embedding provider to use.",
    )
    # Changed default model to generic 'text-embedding-3-small'
    # but the user should override this if using huggingface (e.g., 'all-MiniLM-L6-v2')
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-small",
        help="The model name (e.g. 'text-embedding-3-small' for OpenAI or 'all-MiniLM-L6-v2' for HF)",
    )
    parser.add_argument("--db_name", type=str, default="db")
    parser.add_argument("--collection_name", type=str, default="documents")
    parser.add_argument("--batch_size", type=int, default=256)

    return parser.parse_args()


def ingest_corpus(
    corpus_file_path: str,
    embedding_provider: str,
    embedding_model: str,
    db_name: str,
    collection_name: str,
    batch_size: int,
):
    print(
        f"Initializing {embedding_provider} embeddings with model: {embedding_model}..."
    )

    # Logic to switch providers
    if embedding_provider == "openai":
        embeddings = OpenAIEmbeddings(model=embedding_model)
    elif embedding_provider == "huggingface":
        # Ensure you have sentence_transformers installed: pip install sentence_transformers
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    else:
        raise ValueError(f"Unsupported provider: {embedding_provider}")

    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=db_name,
        embedding_function=embeddings,
        collection_metadata={
            "embedding_model": embedding_model,
            "provider": embedding_provider,
        },
    )

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[dict] = []

    # Count lines for tqdm
    try:
        with open(corpus_file_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
    except FileNotFoundError:
        print(f"Error: File not found at {corpus_file_path}")
        sys.exit(1)

    # Ingest
    with open(corpus_file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Ingesting documents", total=total_lines):

            if not line.strip():
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = str(doc.get("_id", hash(line)))  # Fallback if _id is missing
            merged_text = f"{doc.get('title', '')}\n\n{doc.get('text', '')}"

            ids.append(doc_id)
            documents.append(merged_text)
            metadatas.append({"title": doc.get("title", "")})

            if len(ids) >= batch_size:
                vectordb.add_texts(
                    texts=documents,
                    metadatas=metadatas,
                    ids=ids,
                )
                ids, documents, metadatas = [], [], []

    if ids:
        vectordb.add_texts(
            texts=documents,
            metadatas=metadatas,
            ids=ids,
        )

    print("Ingestion complete.")
    collection = vectordb._collection
    count = collection.count()

    # Simple check to see dimensions
    try:
        sample_embedding = vectordb.get(limit=1, include=["embeddings"])["embeddings"][
            0
        ]
        dimensions = len(sample_embedding)
        print(
            f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store"
        )
    except IndexError:
        print("Vector store is empty.")


if __name__ == "__main__":
    # loads the key from the env (requires an OpenAI key if using OpenAI)
    load_dotenv(override=True)
    args = parse_args()

    ingest_corpus(
        corpus_file_path=args.corpus_file_path,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        db_name=args.db_name,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
    )
