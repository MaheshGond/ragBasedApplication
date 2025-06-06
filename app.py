import streamlit as st
from pathlib import Path
import tempfile
import os
from ingestion.backend_ingest import ingest_pdf_to_qdrant
from chats.rag_query import query_pdf_context

st.title("📄 Ask Your PDF")

if "file_uploaded" not in st.session_state:
    st.session_state["file_uploaded"] = False
if "ingested" not in st.session_state:
    st.session_state["ingested"] = False

uploaded_file = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    # If new file or no ingestion done yet
    if (not st.session_state["file_uploaded"]) or (uploaded_file.name != st.session_state.get("uploaded_filename", "")):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp.flush()
            temp_path = Path(tmp.name)

        st.session_state["file_uploaded"] = True
        st.session_state["pdf_path"] = temp_path
        st.session_state["uploaded_filename"] = uploaded_file.name
        st.session_state["ingested"] = False  # reset ingestion flag

if st.session_state["file_uploaded"] and not st.session_state["ingested"]:
    st.info("Indexing PDF, please wait...")
    ingest_pdf_to_qdrant(st.session_state["pdf_path"], collection_name="session_collection")
    st.session_state["ingested"] = True
    st.success("✅ Document indexed successfully.")

# Ask questions only after ingestion
if st.session_state.get("ingested", False):
    user_query = st.text_input("Ask a question about the PDF")
    if user_query:
        with st.spinner("Thinking..."):
            response = query_pdf_context(user_query, collection_name="session_collection")
            st.markdown(f"**🤖 Answer:**\n\n{response}")

# Reset session button
if st.button("🔁 Reset Session"):
    try:
        if "pdf_path" in st.session_state:
            os.remove(st.session_state["pdf_path"])
    except Exception:
        pass
    st.session_state.clear()
    st.experimental_rerun()
