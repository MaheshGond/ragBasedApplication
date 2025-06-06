import time
import logging
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
import pdfplumber
import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="🔹 [%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_pdf_page(pdf_path: Path, page_index: int, page_data) -> Document | None:
    """Extract text + tables from a single PDF page."""
    try:
        text = page_data.extract_text() or ""
        tables = page_data.extract_tables()
        table_texts = []

        for table in tables:
            rows = [
                " ".join(cell if cell is not None else "" for cell in row)
                for row in table if row
            ]
            table_texts.append("\n".join(rows))

        combined = text + "\n\n" + "\n\n".join(table_texts)
        if combined.strip():
            return Document(
                page_content=combined.strip(),
                metadata={"source": str(pdf_path), "page": page_index + 1}
            )
    except Exception as e:
        logger.error(f"Failed processing page {page_index + 1}: {e}")
    return None


def extract_pdf_text_and_tables(pdf_path: Path) -> List[Document]:
    """Extract all pages from a PDF in parallel."""
    logger.info(f"📄 Starting extraction from {pdf_path.name}...")
    start = time.perf_counter()

    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_pdf_page, pdf_path, i, page)
                for i, page in enumerate(pdf.pages)
            ]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    docs.append(result)

    end = time.perf_counter()
    logger.info(f"✅ Extracted {len(docs)} pages in {end - start:.2f}s")
    return docs


def ingest_pdf_to_qdrant(pdf_path: Path, collection_name: str):
    logger.info(f"📥 Ingesting PDF: {pdf_path.name}")
    overall_start = time.perf_counter()

    # Step 1: Init embeddings and detect vector size dynamically
    embedding_model = embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
        )
    sample_vector = embedding_model.embed_query("test")
    vector_dim = len(sample_vector)
    logger.info(f"🔢 Embedding vector dimension: {vector_dim}")

    # Step 2: Setup Qdrant client and recreate collection
    client = QdrantClient(
                        url="https://68d34727-917a-499c-9bb0-11fa8d527f1d.us-west-1-0.aws.cloud.qdrant.io:6333",
                        api_key=st.secrets["QDRANT_API_KEY"],
                        timeout=40)
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name in existing_collections:
        logger.info(f"🗑️ Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name=collection_name)

    logger.info(f"📦 Creating new collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
    )

    # Step 3: Extract PDF pages
    raw_docs = extract_pdf_text_and_tables(pdf_path)

    # ⚠️ SKIP splitting to keep full-page context
    split_docs = raw_docs

    # Step 4: Generate embeddings and full payloads
    logger.info("🧠 Generating embeddings...")
    texts = [doc.page_content for doc in split_docs]
    vectors = embedding_model.embed_documents(texts)
    payloads = [
        {
            **doc.metadata,
            "text": doc.page_content,
            "page_content": doc.page_content
        }
        for doc in split_docs
    ]

    # Step 5: Upload to Qdrant
    logger.info("🚀 Uploading vectors to Qdrant...")
    client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        payload=payloads,
        ids=None,
        batch_size=64,
        parallel=4
    )

    overall_end = time.perf_counter()
    logger.info(f"🎉 Ingestion complete! Total time: {overall_end - overall_start:.2f}s")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python backend_ingest.py <path_to_pdf> <qdrant_collection_name>")
        sys.exit(1)
    pdf_path = Path(sys.argv[1])
    collection_name = sys.argv[2]
    ingest_pdf_to_qdrant(pdf_path, collection_name)
