from pathlib import Path
from typing import Any, List, Optional, Tuple

import streamlit as st

try:
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document  # type: ignore

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    LANGCHAIN_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - graceful degradation
    Document = Any  # type: ignore[assignment]
    RecursiveCharacterTextSplitter = PyPDFLoader = TextLoader = FAISS = OpenAIEmbeddings = None  # type: ignore
    LANGCHAIN_AVAILABLE = False


def load_internal_documents(doc_dir: Path) -> Tuple[List[Document], List[str]]:
    documents: List[Document] = []
    errors: List[str] = []

    if not LANGCHAIN_AVAILABLE:
        errors.append(
            "LangChain extras are missing. Install dependencies with "
            "`pip install -r requirements.txt`."
        )
        return documents, errors

    if not doc_dir.exists():
        return documents, errors

    for path in sorted(doc_dir.glob("**/*")):
        if path.is_dir():
            continue

        loader: Optional[Any] = None
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            continue

        try:
            documents.extend(loader.load())  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - user-provided docs can be messy
            errors.append(f"{path.name}: {exc}")

    return documents, errors


@st.cache_resource(show_spinner="Indexing documents for retrieval...")
def build_vector_store(
    api_key: str,
    doc_dir: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[Optional["FAISS"], List[str], int]:
    if not LANGCHAIN_AVAILABLE:
        return None, [
            "LangChain components are not installed. "
            "Run `pip install -r requirements.txt` to enable knowledge retrieval."
        ], 0

    docs_path = Path(doc_dir)
    docs_path.mkdir(parents=True, exist_ok=True)

    documents, errors = load_internal_documents(docs_path)
    if not documents:
        return None, errors, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(documents)
    if not split_docs:
        return None, errors, 0

    try:
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=api_key,
        )
        vector_store = FAISS.from_documents(split_docs, embeddings)
    except Exception as exc:  # pragma: no cover - API/network failure
        errors.append(f"Embedding error: {exc}")
        return None, errors, 0

    return vector_store, errors, len(split_docs)


def retrieve_documents(
    vector_store: Optional["FAISS"],
    prompt: str,
    k: int,
) -> Tuple[str, List[Document], Optional[str]]:
    if vector_store is None or not LANGCHAIN_AVAILABLE:
        return "", [], None

    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        if hasattr(retriever, "get_relevant_documents"):
            documents = retriever.get_relevant_documents(prompt)  # type: ignore[attr-defined]
        else:
            result = retriever.invoke(prompt)
            if isinstance(result, list):
                documents = result
            elif result is not None:
                documents = [result]  # type: ignore[list-item]
            else:
                documents = []
    except Exception as exc:
        return "", [], str(exc)

    context = "\n\n".join(doc.page_content.strip() for doc in documents if doc.page_content)
    return context, documents, None
