
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "medical_docs"
INDEX_DIR = PROJECT_ROOT / "vectorstore" / "faiss_index"

EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL",
    "gemini-embedding-001",
)


def fix_symbol_font_text(text):
    """Decode PDFs (for example "Microsoft Print to PDF") that store letters as private-use
    symbols U+F020..U+F0FF. Subtracting 0xF000 restores the real ASCII characters."""
    return "".join(
        chr(ord(ch) - 0xF000) if 0xF020 <= ord(ch) <= 0xF0FF else ch
        for ch in (text or "")
    )


def get_embeddings():
    """Create the configured Google embedding model."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY missing. Add it to your .env file."
        )

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )


def build_faiss_index():
    """Read medical PDFs, split text, and rebuild the FAISS index."""
    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"Medical PDF folder not found: {PDF_DIR}"
        )

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in {PDF_DIR}."
        )

    documents = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_file.name
            page.page_content = fix_symbol_font_text(page.page_content)

        documents.extend(pages)

    documents = [
        doc for doc in documents
        if doc.page_content.strip()
    ]

    if not documents:
        raise ValueError(
            "No extractable text found in the PDFs. "
            "Scanned PDFs may require OCR."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks could be created.")

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    return len(pdf_files), len(chunks)


def load_faiss_index():
    """Load the locally generated, trusted FAISS index."""
    index_file = INDEX_DIR / "index.faiss"

    if not index_file.exists():
        raise FileNotFoundError(
            "FAISS index not found. Build the knowledge base first."
        )

    embeddings = get_embeddings()

    # Only load index files generated and controlled by this project.
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_medical_context(
    question,
    k=4,
    score_threshold=0.55,
):
    """
    Retrieve relevant medical passages.

    Relevance scores range from 0 to 1 in LangChain's relevance-score
    interface; higher scores indicate greater relevance.
    """
    if not question or not question.strip():
        return []

    vectorstore = load_faiss_index()

    results = vectorstore.similarity_search_with_relevance_scores(
        question.strip(),
        k=k,
    )

    passages = []
    seen = set()

    for doc, score in results:
        score = float(score)

        # Ignore passages below the configured relevance threshold.
        if score < score_threshold:
            continue

        text = doc.page_content.strip()
        source = doc.metadata.get("source", "Unknown PDF")
        page = int(doc.metadata.get("page", 0)) + 1

        # Remove repeated copies of the same passage.
        key = (source, page, text)

        if key in seen:
            continue

        seen.add(key)

        passages.append({
            "text": text,
            "source": source,
            "page": page,
            "relevance_score": round(score, 3),
        })

    return passages
