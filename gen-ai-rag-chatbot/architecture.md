# Architecture

This project implements a compact retrieval-augmented generation pipeline for question answering over uploaded PDFs.

## Pipeline

1. The user uploads a PDF in the Gradio UI.
2. LangChain's `PyPDFLoader` extracts page text.
3. `RecursiveCharacterTextSplitter` breaks the document into overlapping chunks.
4. Vertex AI embeddings convert chunks into vectors.
5. Chroma stores the vectors and retrieves semantically similar chunks for a user question.
6. LangChain's `RetrievalQA` chain passes the retrieved context to Gemini on Vertex AI.
7. The answer is returned in the Gradio interface.

## Main Files

- `app/main.py`: starts the local Gradio app.
- `app/ui.py`: defines the upload and question-answering interface.
- `app/rag_pipeline.py`: contains the RAG workflow.
- `app/config.py`: centralizes environment-based configuration.

## Design Choices

- Chroma is used as an in-memory vector store for a simple local demo.
- The vector index is rebuilt per upload so each question is grounded in the selected PDF.
- Configuration is kept outside the code through environment variables.
