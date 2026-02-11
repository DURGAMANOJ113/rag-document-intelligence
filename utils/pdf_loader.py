"""
Utility for loading PDF files into LangChain `Document` objects.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(pdf_path: str | Path) -> List[Document]:
    """
    Load a PDF from disk and return a list of LangChain `Document` objects.

    Each page will be a separate `Document` with associated metadata.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of `Document` instances, one per page.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found at: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()
    return documents

