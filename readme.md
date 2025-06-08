
# RAG-Based PDF Chatbot 📄🤖

A Streamlit-based application that allows users to upload PDFs and interact with them using a chatbot powered by Retrieval-Augmented Generation (RAG). Perfect for decoding large documents like product manuals, contracts, or research papers — with the help of LLMs and vector databases.

---

## 🔍 Features

- Upload and parse PDFs.
- Chunk and embed text for vector storage.
- Query the PDF context using GPT-4 + RAG.
- Beautiful Streamlit chat interface.
- Deploy-ready on Streamlit Cloud.

---

## 🎓 Tech Stack

- **Streamlit** for UI
- **pdfplumber** for PDF parsing
- **LangChain** for RAG pipeline
- **Qdrant** as vector DB
- **OpenAI GPT-4** as the language model
- **dotenv** for config

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone git@github.com:MaheshGond/ragBasedApplication.git
cd ragBasedApplication
```

### 2. Create virtual environment & activate
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create .env file
```env
OPENAI_API_KEY=your_openai_key
QDRANT_API_KEY=your_qdrant_key
QDRANT_URL=https://your-qdrant-url.cloud.qdrant.io
```

---

## 📷 Folder Structure
```text
.
├── app.py
├── chats
│   └── rag_query.py
├── ingestion
│   └── backend_ingest.py
├── requirements.txt
├── .env
└── README.md
```

---

## 🔹 How It Works

### 🔍 Ingestion Pipeline
1. Upload a PDF.
2. Use `pdfplumber` to extract text.
3. Split into chunks (based on tokens).
4. Generate embeddings using OpenAI Embeddings.
5. Store them in Qdrant.

### 🤖 RAG in Action
1. User asks a question in chat.
2. Vector similarity search finds relevant chunks.
3. Context is built and passed to GPT-4.
4. Response is generated based on actual document context.

---

## 🚀 Run Locally
```bash
streamlit run app.py
```

---

## 🌐 Deploy to Streamlit Cloud
1. Push your repo to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Connect GitHub and select the repo.
4. Set environment variables under `Advanced settings`.
5. Click **Deploy**.

---

## 🧐 Example Usage
Ask questions like:
- "Summarize the contract in 5 points."
- "What are the key responsibilities mentioned on page 14?"
- "When is the warranty expiring?"

---

## 🚜 Future Improvements
- Support for multi-PDF context
- UI enhancements
- PDF download of chat transcript

---

## 🌟 Credits
Built with love by [Mahesh Gond](https://github.com/MaheshGond) and Open Source contributors.

---

## 🚀 Related Hashtags
`#AI`, `#OpenAI`, `#LangChain`, `#Qdrant`, `#VectorSearch`, `#PDFtoChatbot`, `#Streamlit`, `#LLM`, `#GPT4`, `#GenerativeAI`

---

## 🔗 License
MIT License
