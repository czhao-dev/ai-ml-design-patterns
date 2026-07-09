"""Tests for app/ui.py."""

import gradio as gr

from app.ui import build_interface


def test_build_interface_returns_a_gradio_interface():
    app = build_interface()
    assert isinstance(app, gr.Interface)


def test_build_interface_accepts_all_advertised_document_types():
    app = build_interface()
    file_input = app.input_components[0]
    assert set(file_input.file_types) == {".pdf", ".txt", ".md", ".csv", ".docx"}


def test_build_interface_has_one_text_output():
    app = build_interface()
    assert len(app.output_components) == 1
    assert isinstance(app.output_components[0], gr.Textbox)
