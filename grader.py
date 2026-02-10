from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import pandas as pd
import json
import time
import re

# ---------- config ----------
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "responses" / "Q1_BASE_1.csv"
OUTPUT_CSV = BASE_DIR / "responses" / "Q1_BASE_1_GRADED.csv"
REFERENCE_ANSWERS = BASE_DIR /'QA'/ "A1.txt"
MODEL = "gpt-5"
# ----------------------------

GRADING_GUIDELINES = """
You are grading answers to vignette case questions.

Compare the student's answer against the reference answer.

Score each answer discretely as:
- 1 (correct): The student's answer contains the key factual information from the reference answer, even if worded differently
- 0.5 (partially correct): The student's answer contains some correct information but is incomplete or contains some inaccuracies
- 0 (wrong): The student's answer contradicts the reference answer or is completely incorrect

Focus on factual accuracy and completeness, not style or exact wording.
Be strict but fair.
"""

def safe_parse_json(text):
    """Extract and parse the first JSON object from the model's text."""
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None

def make_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment or .env file")
    return OpenAI(api_key=api_key)

def load_reference_answers(filepath):
    """Load reference answers from the text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by "A: " and remove the first empty element
    answers = [a.strip() for a in content.split('A: ') if a.strip()]
    return answers

def grade_one_qa(client, question: str, student_answer: str, reference_answer: str) -> tuple[int, str]:
    """
    Send one Question+Student Answer+Reference Answer to the model, get back (score, explanation).
    """
    prompt = f"""
You are an expert grader. Always return ONLY valid JSON – no markdown, no extra text.

Rubric:
{GRADING_GUIDELINES}

Question:
{question}

Reference answer (correct answer from guidelines):
{reference_answer}

Student answer:
{student_answer}

Compare the student's answer to the reference answer and determine if it is factually correct.

Return a JSON object with exactly these fields:
- "score": a number from 0, 0.5, or 1
- "explanation": a short 1-3 sentence explanation for the score, starting with either CORRECT, PARTIALLY CORRECT, or WRONG

Only output valid JSON, nothing else.
"""

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    )

    text = response.output_text.strip()

    data = safe_parse_json(text)
    if data and "score" in data:
        score = float(data["score"])
        explanation = str(data.get("explanation", "")).strip()
    else:
        score = 0
        explanation = f"Could not parse model response: {text}"

    return score, explanation

def main():
    client = make_client()
    
    # Load input CSV
    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
    
    if "Q" not in df.columns or "A" not in df.columns:
        raise ValueError("Input CSV must have columns named 'Q' and 'A'.")
    
    # Load reference answers
    reference_answers = load_reference_answers(REFERENCE_ANSWERS)
    
    # Ensure we have the same number of reference answers as questions
    if len(reference_answers) != len(df):
        print(f"Warning: {len(reference_answers)} reference answers but {len(df)} questions in CSV")
        # Pad with empty strings if needed
        while len(reference_answers) < len(df):
            reference_answers.append("")
    
    scores = []
    explanations = []

    for idx, row in df.iterrows():
        q = str(row["Q"])
        a = str(row["A"])
        
        # Skip empty rows
        if pd.isna(q) or q.strip() == "" or pd.isna(a) or a.strip() == "":
            scores.append(0)
            explanations.append("WRONG. Empty question or answer.")
            continue
        
        # Get corresponding reference answer
        ref_answer = reference_answers[idx] if idx < len(reference_answers) else ""
        
        if not ref_answer:
            scores.append(0)
            explanations.append("WRONG. No reference answer available.")
            continue
        
        print(f"Grading row {idx+1}/{len(df)}...")
        
        score, explanation = grade_one_qa(client, q, a, ref_answer)
        scores.append(score)
        explanations.append(explanation)
        
        time.sleep(0.5)

    df["score"] = scores
    df["explanation"] = explanations

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved scored CSV to: {OUTPUT_CSV}")
    
    # Print summary statistics
    total_score = sum(scores)
    max_score = len([s for s in scores if s >= 0])
    avg_score = total_score / max_score if max_score > 0 else 0
    print(f"\nSummary:")
    print(f"Total score: {total_score}/{max_score}")
    print(f"Average score: {avg_score:.2%}")
    print(f"Correct (1.0): {sum(1 for s in scores if s == 1.0)}")
    print(f"Partially correct (0.5): {sum(1 for s in scores if s == 0.5)}")
    print(f"Wrong (0.0): {sum(1 for s in scores if s == 0.0)}")

if __name__ == "__main__":

    main()
