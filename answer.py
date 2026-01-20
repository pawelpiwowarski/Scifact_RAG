from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential

load_dotenv(override=True)


# 2. Configuration for Groq


DB_NAME = "db"
RETRIEVAL_K = 20
RETRIEVAL_AFTER_RERANK_K = 10


chat_model = "moonshotai/kimi-k2-instruct-0905"
llm = ChatGroq(temperature=0, model_name=chat_model)

# Embeddings (kept as HuggingFace per your snippet)
embedding_model = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings,
    collection_name="documents",
)
retriever = vectorstore.as_retriever()

# Ensure GROQ_API_KEY is in your .env file


reranker_model = "openai/gpt-oss-120b"
# reranker_model = "gpt-5-nano"


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of documents, from most relevant to least relevant, by document id number"
    )


reranker_llm = ChatGroq(
    temperature=0, model_name=reranker_model
).with_structured_output(RankOrder)
# reranker_llm = ChatOpenAI(
#   temperature=0, model_name=reranker_model
# ).with_structured_output(RankOrder)


def rerank(question, docs):
    reranker_prompt = """
You are a document re-ranker.

The documents are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided documents by relevance to the list question, with the most relevant question first.
Reply only with the list of ranked document ids, nothing else. Include all the document ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the documents by relevance to the question, from most relevant to least relevant. Include all the document ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the documents:\n\n"
    for index, doc in enumerate(docs):
        user_prompt += f"# DOCUMENT ID: {index + 1}:\n\n{doc.page_content}\n\n"
    user_prompt += f"Reply only with the list of ranked document ids, nothing else. Return them as valid JSON matching this schema:{RankOrder.model_json_schema()}"

    response = reranker_llm.invoke(
        input=[
            SystemMessage(content=reranker_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    order = response.order
    return [docs[i - 1] for i in order]


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    # Note: Chroma retriever doesn't accept 'k' in invoke directly,
    # it is usually set in as_retriever(search_kwargs={"k": k})
    # But for compatibility with your snippet, we assume the retriever is configured correctly.
    # To be safe, we can configure k here if needed:
    retriever.search_kwargs["k"] = RETRIEVAL_K
    return retriever.invoke(question)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def stream_answer_question(
    question: str, history: list[dict] = [], rerank_docs: bool = True
):
    """
    Generator function that yields chunks of the answer.
    Returns: (chunk, docs)
    """
    main_prompt = """
        You are a knowledgeable, friendly assistant who answers questions about scientific documents.
        You provide clear, concise, and accurate answers based on the provided context.
        If relevant, use the given context to answer any question.
        If you don't know the answer, say so.
        Context:
        {context}
        """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    if rerank_docs:
        docs = rerank(combined, docs)
    docs = docs[:RETRIEVAL_AFTER_RERANK_K]

    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = main_prompt.format(context=context)

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))

    # Stream the response
    for chunk in llm.stream(messages):
        yield chunk.content, docs
