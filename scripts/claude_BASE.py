import csv
from openai import OpenAI
from dotenv import load_dotenv
from anthropic import Anthropic
from pathlib import Path
import os

# ---------- config ----------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
QUESTIONS_FILE = PROJECT_ROOT / "QA" / "Q1.txt"
OUTPUT_FILE = PROJECT_ROOT / "responses" / "claude_Q1_BASE_3.csv"
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

def build_prompt(question_text: str) -> str:
    return f"Question: {question_text}".strip()

def main():
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file")

    client = Anthropic(api_key=api_key)

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

        msg = client.messages.create(
            model=MODEL,
            max_tokens=512,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(question)}],
        )

        # Anthropic returns a list of content blocks; most often the first is text
        answer = ""
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                answer += block.text
        answer = answer.strip()

        rows.append([question, answer])

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Q", "A"])
        writer.writerows(rows)

    print(f"\nCSV written to {OUTPUT_FILE}")
    print(f"Total questions processed: {len(rows)}")

if __name__ == "__main__":
    main()