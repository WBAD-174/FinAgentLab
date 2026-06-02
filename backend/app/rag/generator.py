from typing import Dict, List

from openai import OpenAI

from backend.app.core.config import settings


client = OpenAI(api_key=settings.openai_api_key)


SYSTEM_PROMPT = """
You are a careful RAG assistant.

Use only the provided context to answer the user's question.
If the context does not contain enough information, say you don't know.
Always cite the source filename and chunk index when using evidence.

Answer in the same language as the user's question.
"""


def build_context(chunks: List[Dict]) -> str:
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"""
[Context {index}]
Source: {chunk["source"]}
Chunk index: {chunk["chunk_index"]}
Text:
{chunk["text"]}
"""
        )

    return "\n".join(context_parts)


def generate_answer(question: str, chunks: List[Dict]) -> str:
    context = build_context(chunks)

    user_prompt = f"""
Context:
{context}

Question:
{question}

Answer with citations:
"""

    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content