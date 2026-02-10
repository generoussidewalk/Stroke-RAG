import csv
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

# ---------- config ----------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
QUESTIONS_FILE = PROJECT_ROOT / "QA" / "Q1.txt"
OUTPUT_FILE = PROJECT_ROOT / "responses" / "Q1_BASE_2.csv"
MODEL = "gpt-5"
# -----------------------------


SYSTEM_PROMPT = """
You are an expert acute ischemic stroke answering assistant.
Follow any instructions given in the user message exactly, especially:
- Answer directly.
- Answer in one sentence.
- Use complete sentences with rationale.
- Always provide an answer.
""".strip()

def build_prompt(question_text):
    """
    Build a prompt for a single question.
    """
    prompt = f"""

    Question: {question_text}
    """
    return prompt.strip()


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")

    client = OpenAI(api_key=api_key)

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

        response = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": build_prompt(question)}],
                },
            ],
        )

        answer = response.output_text.strip()
        rows.append([question, answer])

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Q", "A"])
        writer.writerows(rows)

    print(f"\nCSV written to {OUTPUT_FILE}")
    print(f"Total questions processed: {len(rows)}")


if __name__ == "__main__":
    main()