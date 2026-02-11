"""
Flask RAG Document Intelligence System
Using:
- LangChain
- FAISS
- Local LLM via Ollama
"""

import markdown
from pathlib import Path
from typing import List, Tuple

from flask import Flask, render_template, request, redirect, url_for, flash

from langchain_core.documents import Document
from langchain_community.llms import Ollama

from utils.pdf_loader import load_pdf
from utils.vector_store import split_documents, build_faiss_index


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "dev-secret-key"

VECTOR_STORE = None
CURRENT_SOURCE_DOCS: List[Document] = []


def build_index_for_pdf(pdf_path: Path):
    raw_documents = load_pdf(pdf_path)
    chunked_documents = split_documents(raw_documents)
    vector_store, _ = build_faiss_index(chunked_documents)
    return vector_store, chunked_documents


def answer_question(question: str, k: int = 3) -> Tuple[str, List[Document]]:
    global VECTOR_STORE

    if VECTOR_STORE is None:
        raise RuntimeError("Upload a PDF first.")

    # Retrieve top-k similar chunks
    similar_docs: List[Document] = VECTOR_STORE.similarity_search(question, k=k)

    context_text = "\n\n".join(
        f"Chunk {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(similar_docs)
    )

    # Local LLM using Ollama
    llm = Ollama(model="llama3")  # Make sure llama3 is pulled

    prompt = f"""
    You are an expert AI research assistant.

    Your task:
    - Answer the question in a detailed, clear, and well-structured manner.
    - Use ONLY the provided context.
    - If information is missing, clearly say so.
    - Explain concepts step-by-step if needed.
    - Use headings or bullet points when appropriate.
    - Avoid generic summaries.
    - Focus directly on the user's question.

    Context:
    {context_text}

    User Question:
    {question}

    Provide a comprehensive answer:
    """


    answer = llm.invoke(prompt)

    return answer, similar_docs


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        answer=None,
        question=None,
        source_docs=[],
    )


@app.route("/upload", methods=["POST"])
def upload():
    global VECTOR_STORE, CURRENT_SOURCE_DOCS

    file = request.files.get("pdf_file")

    if file is None or file.filename == "":
        flash("Please select a PDF file.", "error")
        return redirect(url_for("index"))

    if not file.filename.lower().endswith(".pdf"):
        flash("Only PDF files allowed.", "error")
        return redirect(url_for("index"))

    save_path = UPLOAD_DIR / file.filename
    file.save(save_path)

    try:
        VECTOR_STORE, CURRENT_SOURCE_DOCS = build_index_for_pdf(save_path)
        flash("PDF indexed successfully!", "success")
    except Exception as exc:
        flash(f"Error processing PDF: {exc}", "error")
        VECTOR_STORE = None
        CURRENT_SOURCE_DOCS = []

    return redirect(url_for("index"))


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()

    if not question:
        flash("Please enter a question.", "error")
        return redirect(url_for("index"))

    try:
        answer, source_docs = answer_question(question, k=3)
        answer = markdown.markdown(answer)
    except Exception as exc:
        flash(f"Error: {exc}", "error")
        return redirect(url_for("index"))

    return render_template(
        "index.html",
        answer=answer,
        question=question,
        source_docs=source_docs,
    )


if __name__ == "__main__":
    app.run(debug=True)
