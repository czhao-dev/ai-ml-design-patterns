"""Examples of splitting text for retrieval-augmented generation."""

from __future__ import annotations

from pathlib import Path

from langchain.text_splitter import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from langchain_text_splitters import HTMLHeaderTextSplitter, HTMLSectionSplitter


ROOT = Path(__file__).resolve().parents[1]
POLICY_FILE = (
    ROOT / "data" / "sample_documents" / "company_policies.txt"
)


def load_policy_text() -> str:
    return POLICY_FILE.read_text()


def print_result(label: str, docs, limit: int = 220) -> None:
    print(f"\n{label}: {len(docs)} chunk(s)")
    if docs:
        content = getattr(docs[0], "page_content", docs[0])
        print(content[:limit].replace("\n", " "))


def split_by_character(text: str):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
    )
    return splitter.create_documents([text], metadatas=[{"document": "Company Policies"}])


def split_recursively(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
    )
    return splitter.create_documents([text])


def split_python_code():
    code = """
def hello_world():
    print("Hello, World!")

hello_world()
"""
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=50,
        chunk_overlap=0,
    )
    return splitter.create_documents([code])


def split_javascript_code():
    code = """
function helloWorld() {
  console.log("Hello, World!");
}

helloWorld();
"""
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS,
        chunk_size=60,
        chunk_overlap=0,
    )
    return splitter.create_documents([code])


def split_markdown_headers():
    markdown = "# Foo\n\n## Bar\n\nHi this is Jim\n\n### Boo\n\nHi this is Lance\n\n## Baz\n\nHi this is Molly"
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )
    return splitter.split_text(markdown)


def split_html_headers():
    html = """
<!DOCTYPE html>
<html>
  <body>
    <h1>Foo</h1>
    <p>Some intro text about Foo.</p>
    <h2>Bar main section</h2>
    <p>Some intro text about Bar.</p>
    <h3>Bar subsection 1</h3>
    <p>Some text about the first subtopic of Bar.</p>
  </body>
</html>
"""
    headers = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]
    return HTMLHeaderTextSplitter(headers_to_split_on=headers).split_text(html)


def split_html_sections():
    html = """
<!DOCTYPE html>
<html>
  <body>
    <h1>Foo</h1>
    <section>
      <h2>Bar main section</h2>
      <p>Some intro text about Bar.</p>
    </section>
  </body>
</html>
"""
    headers = [("h1", "Header 1"), ("h2", "Header 2")]
    return HTMLSectionSplitter(headers_to_split_on=headers).split_text(html)


def split_latex_code():
    latex = r"""
\documentclass{article}
egin{document}
\section{Introduction}
Large language models can generate text and answer questions.
\subsection{Applications}
LLMs are used in chatbots, search, summarization, and assistants.
\end{document}
"""
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.LATEX,
        chunk_size=80,
        chunk_overlap=0,
    )
    return splitter.create_documents([latex])


def main() -> None:
    text = load_policy_text()
    example_doc = Document(
        page_content="Python emphasizes code readability.",
        metadata={"source": "About Python"},
    )
    print(f"Example Document metadata: {example_doc.metadata}")

    print_result("Character splitter", split_by_character(text))
    print_result("Recursive splitter", split_recursively(text))
    print_result("Python code splitter", split_python_code())
    print_result("JavaScript code splitter", split_javascript_code())
    print_result("Markdown header splitter", split_markdown_headers())
    print_result("HTML header splitter", split_html_headers())
    print_result("HTML section splitter", split_html_sections())
    print_result("LaTeX code splitter", split_latex_code())


if __name__ == "__main__":
    main()
