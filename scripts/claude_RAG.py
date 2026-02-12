# claude_RAG.py
from dotenv import load_dotenv
from pathlib import Path
import os
import csv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    PromptTemplate,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_parse import LlamaParse

from llama_index.llms.anthropic import Anthropic


# ---------- config ----------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
QUESTIONS_FILE = PROJECT_ROOT / "QA" / "Q1.txt"
OUTPUT_FILE = PROJECT_ROOT / "responses" / "claude_Q1_RAG_3.csv"
MODEL = "claude-opus-4-6"
# -----------------------------


SYSTEM_PROMPT = """
You are an expert acute ischemic stroke answering assistant.
Follow any instructions given in the user message exactly, especially:
- Answer directly.
- Return exactly ONE sentence (no line breaks).
- Use complete sentences with rationale.
- Always provide an answer.
""".strip()


def main():
    load_dotenv()

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    llama_parse_key = os.getenv("LLAMA_PARSE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file")
    if not llama_parse_key:
        raise ValueError("LLAMA_PARSE_API_KEY not found in .env file")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file (needed for embeddings)")

    llm = Anthropic(
        model=MODEL,
        api_key=anthropic_api_key,
        temperature=0,
        system_prompt=SYSTEM_PROMPT,
    )
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=openai_api_key,
        dimensions=3072,
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 128

    QA_PROMPT_STR = """
Guidelines:
---------------------
{context_str}
---------------------

Question: {query_str}
"""
    qa_prompt = PromptTemplate(QA_PROMPT_STR)

    parser = LlamaParse(
        api_key=llama_parse_key,
        result_type="markdown",
        parsing_instruction=(
            "Extract all clinical content, including descriptions "
            "of any figures, graphs, or tables."
        ),
    )

    documents = SimpleDirectoryReader(
        "./data",
        file_extractor={".pdf": parser},
    ).load_data()

    node_parser = SentenceSplitter(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
        paragraph_separator="\n\n",
    )

    nodes = node_parser.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=20,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.25),
            SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_n=10,
            ),
        ],
    )

    query_engine.update_prompts({
        "response_synthesizer:text_qa_template": qa_prompt
    })

    def run_query(question: str) -> str:
        resp = query_engine.query(question)
        text = getattr(resp, "response", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return str(resp).strip()

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = [
            line.strip().split(":", 1)[1].strip()
            if line.lower().startswith("q:")
            else line.strip()
            for line in f
            if line.strip()
        ]

    rows = []

    for i, question in enumerate(questions, start=1):
        print(f"Processing question {i}/{len(questions)}...")
        answer = run_query(question)
        rows.append([question, answer])

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Q", "A"])
        writer.writerows(rows)

    print(f"\nCSV written to {OUTPUT_FILE}")
    print(f"Total questions processed: {len(rows)}")


if __name__ == "__main__":
    main()
