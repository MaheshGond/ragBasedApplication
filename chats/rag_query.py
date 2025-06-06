import time
import logging
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import streamlit as st
# Load environment and setup logging
import os

# Use Streamlit secrets if available
openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

logging.basicConfig(
    level=logging.INFO,
    format="🔹 [%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def query_pdf_context(user_query: str, collection_name="session_collection") -> str:
    logger.info(f"🔍 Querying context for: '{user_query}'")
    overall_start = time.perf_counter()

    # Load vector store
    t0 = time.perf_counter()
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_db = QdrantVectorStore.from_existing_collection(
        url="https://68d34727-917a-499c-9bb0-11fa8d527f1d.us-west-1-0.aws.cloud.qdrant.io:6333",
        api_key=st.secrets["QDRANT_API_KEY"],
        collection_name=collection_name,
        embedding=embedding_model,
    )
    t1 = time.perf_counter()
    logger.info(f"📚 Loaded Qdrant vector store in {t1 - t0:.2f}s")

    # Similarity search
    t2 = time.perf_counter()
    search_results = vector_db.similarity_search(query=user_query)
    t3 = time.perf_counter()
    logger.info(f"🧭 Retrieved {len(search_results)} relevant chunks in {t3 - t2:.2f}s")

    if not search_results:
        return "⚠️ No relevant content found in the document."

    # Build context
    context = "\n\n".join([
        f"Page Number: {res.metadata.get('page_label') or res.metadata.get('page') or 'Unknown'}\n"
        f"Source: {res.metadata.get('source', 'N/A')}\n"
        f"Content:\n{res.page_content}"
        for res in search_results
    ])

    # Chat completion
    SYSTEM_PROMPT = f"""
You are a helpful assistant that answers user questions based on the following extracted PDF content.

Only use the context provided. Reference page numbers when helpful. Do not guess or fabricate facts.

Context:
{context}
    """

    t4 = time.perf_counter()
    chat_completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_query},
        ]
    )
    t5 = time.perf_counter()
    logger.info(f"🧠 GPT-4.1 response generated in {t5 - t4:.2f}s")

    overall_end = time.perf_counter()
    logger.info(f"✅ Completed query in {overall_end - overall_start:.2f}s")

    return chat_completion.choices[0].message.content.strip()


# Optional CLI testing
if __name__ == "__main__":
    while True:
        query = input("Ask a question > ")
        answer = query_pdf_context(query)
        print(f"\n🤖 {answer}\n")
