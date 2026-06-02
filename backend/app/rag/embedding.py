from typing import List

from openai import OpenAI

from backend.app.core.config import settings


client = OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )

    return [item.embedding for item in response.data]


def embed_query(query: str) -> List[float]:
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=query,
    )

    return response.data[0].embedding