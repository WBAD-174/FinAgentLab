from typing import Dict

from backend.app.rag.generator import generate_answer
from backend.app.rag.retriever import retrieve


def answer_question(question: str) -> Dict:
    chunks = retrieve(question)
    answer = generate_answer(question, chunks)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "distance": chunk["distance"],
                "preview": chunk["text"][:200],
            }
            for chunk in chunks
        ],
    }


if __name__ == "__main__":
    result = answer_question("What is the main idea of this document?")

    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(source)