import argparse
from langchain_chroma import Chroma      
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import time


from embedding import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key="mykey")
    response_text = model.invoke(prompt).content  


    print(f"\n{'='*60}")
    print(f"Question: {query_text}")
    print(f"{'='*60}")
    print(f"\nAnswer:\n{response_text}")
    print(f"\n{'='*60}")
    print("Sources:")
    for i, (doc, score) in enumerate(results):
        chunk_id = doc.metadata.get("id", "unknown")       
        parts = chunk_id.split(":")
        file = parts[0] if len(parts) > 0 else "unknown"
        page = parts[1] if len(parts) > 1 else "unknown"
        chunk_index = parts[2] if len(parts) > 2 else "unknown"
        print(f"\n  [{i+1}] ID: {chunk_id}")
        print(f"       File: {file}")
        print(f"       Page: {int(page)+1}")           
        print(f"       Chunk on page: {chunk_index}")
        print(f"       Similarity score: {score:.4f}")
        print(f"       Content preview: {doc.page_content[:150]}...")
    print(f"{'='*60}\n")

    return response_text


if __name__ == "__main__":
    main()