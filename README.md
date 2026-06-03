# pdf-chat-rag

Chat with any PDF using Chainlit + RAG. Fully free stack.

## Stack

- **Chainlit**: chat interface and file upload
- **LangChain**: RAG orchestration
- **FAISS**: in-memory vector store
- **HuggingFace**: local embeddings (all-MiniLM-L6-v2)
- **Groq**: LLM inference (llama-3.1-8b-instant, free tier)

## Setup

```bash
git clone https://github.com/victorCSA/pdf-chat-rag
cd pdf-chat-rag

cp .env.example .env
# edit .env and add your GROQ_API_KEY

pip install -r requirements.txt
```

## Run

```bash
chainlit run app.py
```

Open http://localhost:8000, upload a PDF, and start asking questions.

## How it works

1. **Indexing**: PDF is split into 500-token chunks, embedded locally
   with all-MiniLM-L6-v2, and stored in a FAISS vector store.
2. **Retrieval**: each question is embedded and the top-4 most similar
   chunks are retrieved by cosine similarity.
3. **Generation**: Llama 3.1 via Groq generates an answer grounded
   in the retrieved chunks. Source pages are shown below each answer.