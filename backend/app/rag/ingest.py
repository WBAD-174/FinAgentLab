import uuid 
from pathlib import Path
from typing import List

import chormadb
from openai import OpenAI
from pypdf import PdfReader

from backend.app.core.config import settings

client = OpenAI(api_key=settings.openai_api_key )

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n[Page {page_index}]\n{text}")

    return "\n".join(pages)

def load_document(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in [".txt", ".md"]:
        return read_txt(path)

    if suffix == ".pdf":
        return read_pdf(path)

    raise ValueError(f"Unsupported file type: {path}")

def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + settings.chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - settings.chunk_overlap

    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )

    return [item.embedding for item in response.data]

def get_collection():
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return chroma_client.get_or_create_collection(
        name=settings.collection_name
    )


def ingest_file(path: Path) -> int:
    print(f"Reading file: {path}")

    text = load_document(path)
    chunks = chunk_text(text)

    if not chunks:
        print(f"No text extracted from: {path}")
        return 0

    embeddings = embed_texts(chunks)
    collection = get_collection()

    ids = []
    metadatas = []

    for chunk_index, _ in enumerate(chunks):
        ids.append(str(uuid.uuid4()))
        metadatas.append(
            {
                "source": path.name,
                "chunk_index": chunk_index,
                "file_path": str(path),
            }
        )

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Ingested {len(chunks)} chunks from {path.name}")
    return len(chunks)


def ingest_all() -> int:
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    for path in settings.raw_data_dir.iterdir():
        if path.is_file() and path.suffix.lower() in [".pdf", ".txt", ".md"]:
            total_chunks += ingest_file(path)

    print(f"Total chunks ingested: {total_chunks}")
    return total_chunks


if __name__ == "__main__":
    ingest_all()