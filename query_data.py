import argparse
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
import sys
import io
# Fix encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5) #5 chunks

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("prompt:", prompt, flush=True)

    model = Ollama(model="mistral")     # or llama3.2:3b plus leger
    #mistral 32k context(can remember 32k tokens) 
    #for exp chatgpt free has approx 8k 
    response_text = model.invoke(prompt)

#found out that as long as embeddings are good miselech to use local llm doesnt really matter 
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    print("Response:", response_text)
    print("Sources:", sources, flush=True)
    return response_text


if __name__ == "__main__":
    main()
