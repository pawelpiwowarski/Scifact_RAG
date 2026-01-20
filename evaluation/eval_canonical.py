from pathlib import Path
import json
import csv
from collections import defaultdict
import math
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm


# Import rerank from the main answer module
from answer import rerank

# Paths
BASE_DIR = Path(__file__).parent.parent
SCIFACT_DIR = BASE_DIR / "scifact"

QUERIES_FILE = SCIFACT_DIR / "queries.jsonl"
QRELS_FILE = SCIFACT_DIR / "qrels" / "test.tsv"

# Vector DB
DB_NAME = "db"
COLLECTION_NAME = "documents"

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Evaluation
K_VALUES = [1, 5, 10]


def load_queries(path):
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries[obj["_id"]] = obj["text"]
    return queries


def load_qrels(path):
    qrels = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for qid, docid, rel in reader:
            if int(rel) > 0:
                qrels[str(qid)].add(str(docid))
    return qrels


def mrr_at_k(ranked_doc_ids, relevant_docs, k):
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0


def dcg(rels):
    return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(rels))


def ndcg_at_k(ranked_ids, relevant_ids, k):
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)

    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_scifact(use_reranker=False):
    """Evaluates retrieval on SciFact canonical test set."""

    queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=DB_NAME,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": max(K_VALUES)})

    mrr_scores = {k: [] for k in K_VALUES}
    ndcg_scores = {k: [] for k in K_VALUES}

    overall_mrr_list = []
    overall_ndcg_list = []

    # Iterate with tqdm
    for qid, query in tqdm(
        queries.items(), total=len(queries), desc="Evaluating queries"
    ):
        if qid not in qrels:
            continue

        # 1. Retrieve
        docs = retriever.invoke(query)

        # 2. Rerank (Optional)
        if use_reranker:
            # We assume rerank returns a list of Documents
            docs = rerank(query, docs)

        ranked_doc_ids = [str(doc.metadata.get("corpus_id", doc.id)) for doc in docs]
        relevant_ids = qrels[qid]

        per_query_mrr = []
        per_query_ndcg = []

        for k in K_VALUES:
            rr = mrr_at_k(ranked_doc_ids, relevant_ids, k)
            ndcg_val = ndcg_at_k(ranked_doc_ids, relevant_ids, k)

            mrr_scores[k].append(rr)
            ndcg_scores[k].append(ndcg_val)

            per_query_mrr.append(rr)
            per_query_ndcg.append(ndcg_val)

        if per_query_mrr:
            overall_mrr_list.append(max(per_query_mrr))
            overall_ndcg_list.append(max(per_query_ndcg))

    avg_mrr = {k: sum(vals) / len(vals) for k, vals in mrr_scores.items()}
    avg_ndcg = {k: sum(vals) / len(vals) for k, vals in ndcg_scores.items()}

    overall_mrr = (
        sum(overall_mrr_list) / len(overall_mrr_list) if overall_mrr_list else 0
    )
    overall_ndcg = (
        sum(overall_ndcg_list) / len(overall_ndcg_list) if overall_ndcg_list else 0
    )

    return {
        "avg_mrr_at_k": avg_mrr,
        "avg_ndcg_at_k": avg_ndcg,
        "overall_mrr": overall_mrr,
        "overall_ndcg": overall_ndcg,
    }
