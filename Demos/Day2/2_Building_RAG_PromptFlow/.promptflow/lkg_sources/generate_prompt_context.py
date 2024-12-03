from typing import List
from promptflow import tool


@tool
def generate_prompt_context(search_result: List) -> str:
    def format_doc(doc: dict):
        return f"Content: {doc['Content']}"

    SOURCE_KEY = "source"
    URL_KEY = "url"

    retrieved_docs = []
    for item in search_result:
        content = item["page_content"]

        retrieved_docs.append({
            "Content": content,
        })

    doc_string = "\n\n".join([format_doc(doc) for doc in retrieved_docs])
    return doc_string
