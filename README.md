---
title: SciFacts Expert Assistant
short_description: Verify scientific claims with RAG
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🧬 SciFacts Expert Assistant

A high-precision **Retrieval-Augmented Generation (RAG)** application designed to verify scientific claims and answer complex biomedical questions using the [SciFacts dataset](https://ir-datasets.com/beir.html#beir/scifact).

This system leverages **LLM-based Reranking** to significantly improve retrieval performance, ensuring the chat model receives the most relevant scientific evidence.

---

![alt text](UI.png "UI of the system")

Check out the Gradio App at https://huggingface.co/spaces/pawlo2013/Scifact_RAG

## ⚡ Technology Stack

| Component           | Technology / Model                   | Why?                                                                         |
| :------------------ | :----------------------------------- | :--------------------------------------------------------------------------- |
| **Frontend UI**     | **Gradio**                           | Interactive web interface with streaming chat and real-time dashboard.       |
| **Orchestration**   | **LangChain**                        | Manages the retrieval chains, prompt templates, and LLM interaction.         |
| **Vector Database** | **ChromaDB**                         | Stores document embeddings for efficient semantic search.                    |
| **Embeddings**      | **HuggingFace** (`all-MiniLM-L6-v2`) | Converts scientific text into 384-dimensional vectors.                       |
| **LLM Provider**    | **Groq**                             | Provides ultra-fast inference for the chat and reranking models.             |
| **Main Model**      | **Kimi-k2-instruct**                 | Handles the final answer synthesis (selected for long-context capabilities). |
| **Reranker**        | **GPT-OSS-120b**                     | Re-ranks retrieved documents to optimize relevance.                          |

---

## 📊 Performance Benchmark: The Impact of Reranking

We evaluated the retrieval system using an **LLM-generated test set** to measure the impact of adding a reranking step.

### 🏆 Retrieval Evaluation Results

| Metric                         | Base Retrieval | With Reranker (GPT-OSS-120b) | Improvement  |
| :----------------------------- | :------------: | :--------------------------: | :----------: |
| **Mean Reciprocal Rank (MRR)** |     0.8193     |          **0.8480**          | 🟢 **+3.5%** |
| **Normalized DCG (nDCG)**      |     0.8079     |          **0.8323**          | 🟢 **+3.0%** |
| **Keyword Coverage**           |     89.3%      |            89.3%             |   ➖ Same    |

> **Insight:** While keyword coverage remained stable, the **Reranker** significantly improved the ranking quality (MRR & nDCG). This means relevant documents are pushed to the top of the context window, reducing hallucinations and improving answer accuracy.

---

## 🏗️ System Architecture

1.  **Ingestion:** The SciFacts corpus is chunked and embedded using `all-MiniLM-L6-v2`.
2.  **Vector Store:** Stored in **ChromaDB** for fast similarity search.
3.  **Retrieval:** Initial fetch of top-k ($k=20$) documents based on cosine similarity.
4.  **Reranking:** The **GPT-OSS-120b** model re-scores the retrieved documents to filter noise, passing only the top ($k=10$) most relevant chunks to the generator.
5.  **Generation:** **Kimi-k2-instruct** synthesizes the final answer based on the refined evidence.

---

## 🚀 Features

- **Interactive UI:** Built with **Gradio**, featuring streaming responses and a side-by-side view of retrieved evidence.
- **Reference Questions:** One-click execution of verified ground-truth questions.
- **Live Evaluation Dashboard:** Built-in dashboard to run and visualize MRR, nDCG, and Answer Accuracy metrics in real-time.
- **Dual Evaluation Modes:**
  - **Canonical:** Standard SciFacts benchmark.
  - **LLM-Generated:** Synthetic test set for broad coverage.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/scifact-rag.git](https://github.com/your-username/scifact-rag.git)
cd scifact-rag
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

_Note: Ensure you have `gradio`, `langchain`, `chromadb`, `pydantic`, and `tiktoken` installed._

### 3. Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # If using OpenAI for evaluation generation
HF_TOKEN = your_hf_token_here #Y ou may also need to login to hugginface or provide a token


```

### 4. Ingest Data (Build Vector DB)

If you haven't built the database yet:

```bash
python ingestion.py --corpus_file_path ./scifact/corpus.jsonl --embedding_provider huggingface

```

### 5. Generate Test Data (Optional)

To create a fresh synthetic test set for evaluation:

```bash
python generate_tests.py --TOTAL_NUMBER_OF_QUESTIONS 50

```

---

## 🖥️ Running the Application

### Main Chat Interface

Launch the research assistant:

```bash
python app.py

```

Access the UI at `http://localhost:7860`

### Evaluation Dashboard

Launch the metrics dashboard to reproduce the benchmark results:

```bash
python dashboard.py

```

---

## 📂 Project Structure

```text
├── app.py                 # Main Gradio Chat Application
├── evaluator.py           # Evaluation Dashboard (Metrics Visualization)
├── answer.py              # Core RAG logic (Retrieval, Reranking, Generation)
├── ingest.py           # Script to load SciFacts into ChromaDB
├── make_test_answers.py      # LLM-based synthetic test generation
├── evaluation/
│   ├── eval.py            # Evaluation logic for Retrieval & Answers
│   ├── eval_canonical.py  # Logic for SciFacts standard benchmark
│   ├── test.py            # Test data loading utilities
│   └── tests.jsonl        # Generated test questions
└── scifact/               # Dataset directory

```

---

## 📜 License

This project is open-source and available under the MIT License.
