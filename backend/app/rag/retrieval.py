from typing import Dict, List

import chromadb
from openai import OpenAI

from backend.app.core.config import settings
from backend.app.rag.embedding import embed_query


client = OpenAI(api_key=settings.openai_api_key)


def get_collection():
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))

    return chroma_client.get_or_create_collection(
        name=settings.collection_name
    )


def retrieve(query: str, top_k: int | None = None) -> List[Dict]:
    if top_k is None:
        top_k = settings.top_k

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    query_embedding = embed_query(query)
    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved_chunks = []

    for text, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}

        retrieved_chunks.append(
            {
                "text": text,
                "source": metadata.get("source"),
                "file_path": metadata.get("file_path"),
                "chunk_index": metadata.get("chunk_index"),
                "distance": distance,
            }
        )

    return retrieved_chunks


if __name__ == "__main__":
    query = "What is this document about?"
    chunks = retrieve(query)

    for index, chunk in enumerate(chunks, start=1):
        print("=" * 80)
        print(f"Result {index}")
        print(f"Source: {chunk['source']}")
        print(f"File path: {chunk['file_path']}")
        print(f"Chunk index: {chunk['chunk_index']}")
        print(f"Distance: {chunk['distance']}")
        print(chunk["text"][:500])