"""Gradio interface for the portfolio RAG application."""

import gradio as gr

from app.rag_pipeline import answer_question


def build_interface() -> gr.Interface:
    return gr.Interface(
        fn=answer_question,
        allow_flagging="never",
        inputs=[
            gr.File(
                label="Upload Document",
                file_count="single",
                file_types=[".pdf", ".txt", ".md", ".csv", ".docx"],
                type="filepath",
            ),
            gr.Textbox(
                label="Question",
                lines=2,
                placeholder="Ask a question about the uploaded document...",
            ),
        ],
        outputs=gr.Textbox(label="Answer"),
        title="RAG Document QA Chatbot",
        description=(
            "Upload a document (PDF, TXT, Markdown, CSV, or DOCX) and ask questions "
            "answered from retrieved document context."
        ),
    )
