# import csv
# import os
# import re
# import time
# from pathlib import Path
# from dotenv import load_dotenv
# from google import genai
# from google.genai import types

# # ---------- config ----------
# BASE_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = BASE_DIR.parent
# QUESTIONS_FILE = PROJECT_ROOT / "QA" / "Q2.txt"
# OUTPUT_FILE = PROJECT_ROOT / "responses" / "gemini_Q2_BASE_fixed.csv"
# MODEL = "gemini-2.5-flash"
# MAX_OUTPUT_TOKENS = 1024
# # -----------------------------

# SYSTEM_PROMPT = """
# You are an expert acute ischemic stroke answering assistant.
# Follow any instructions given in the user message exactly, especially:
# - Answer directly.
# - Return exactly ONE sentence (no line breaks).
# - Use complete sentences with rationale.
# - Always provide an answer.
# """.strip()

# CONCISE_SYSTEM_PROMPT = """
# You are an expert acute ischemic stroke answering assistant.
# Follow any instructions given in the user message exactly, especially:
# - Answer directly.
# - Return exactly ONE sentence (no line breaks).
# - Use complete sentences with rationale.
# - Always provide an answer.
# """.strip()


# def build_prompt(question_text: str) -> str:
#     return f"Question: {question_text}".strip()


# def extract_text_and_finish_reason(response):
#     answer = ""
#     finish_reason = "UNKNOWN"

#     candidates = getattr(response, "candidates", None) or []
#     if not candidates:
#         # FIX 3: No candidates usually means a prompt-level safety block.
#         # Return a distinct marker so the caller can log/handle it.
#         return "[BLOCKED: no candidates returned]", "SAFETY_BLOCK"

#     cand = candidates[0]
#     finish_reason = str(getattr(cand, "finish_reason", "UNKNOWN"))

#     # FIX 4: Explicitly detect SAFETY finish reason before trying to read content.
#     norm = normalize_finish_reason(finish_reason)
#     if norm == "SAFETY":
#         return "[BLOCKED: finish_reason=SAFETY]", "SAFETY"

#     content = getattr(cand, "content", None)
#     parts = getattr(content, "parts", None) or []

#     text_chunks = []
#     for part in parts:
#         text = getattr(part, "text", None)
#         if text:
#             text_chunks.append(text)

#     answer = "".join(text_chunks).strip()
#     return answer, finish_reason


# def normalize_finish_reason(finish_reason: str) -> str:
#     fr = str(finish_reason).strip().upper()
#     if "." in fr:
#         fr = fr.split(".")[-1]
#     return fr


# def looks_truncated(answer: str) -> bool:
#     if not answer:
#         return True
#     if answer.startswith("[BLOCKED"):
#         return False  # Don't treat safety blocks as truncation

#     answer = answer.strip()

#     # Must end with terminal punctuation
#     if answer[-1] not in ".!?\"'":
#         return True

#     # Ends with a comma (e.g. "bypassing the primary stroke center,")
#     if re.search(r",\s*$", answer):
#         return True

#     # Common dangling conjunctions / prepositions before punctuation
#     # e.g. "...given the." or "...recommended," 
#     unfinished_patterns = [
#         r"\b(and|or|because|with|for|to|of|in|on|if|but|that|which|given|as|the|a|an)\s*[.!?,]?\s*$",
#         r"[,;:\-]\s*$",
#         # Looks like a number was cut: ends in digit or hyphen+digit
#         r"\d[-–]\s*$",
#         r"\d\s*$",
#     ]
#     for pat in unfinished_patterns:
#         if re.search(pat, answer, flags=re.IGNORECASE):
#             return True

#     return False


# def generate_once(client, question, system_prompt):
#     safety_settings = [
#         types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",        threshold="BLOCK_NONE"),
#         types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",          threshold="BLOCK_NONE"),
#         types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",   threshold="BLOCK_NONE"),
#         types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT",   threshold="BLOCK_NONE"),
#         types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY",     threshold="BLOCK_NONE"),
#     ]

#     response = client.models.generate_content(
#         model=MODEL,
#         contents=build_prompt(question),
#         config=types.GenerateContentConfig(
#             system_instruction=system_prompt,
#             temperature=0,
#             max_output_tokens=MAX_OUTPUT_TOKENS,
#             safety_settings=safety_settings,
#         ),
#     )

#     return extract_text_and_finish_reason(response)


# def generate_with_api_retry(client, question, system_prompt, max_retries=5):
#     for attempt in range(max_retries):
#         try:
#             answer, finish_reason = generate_once(client, question, system_prompt)
#             # Don't retry safety blocks — they won't resolve with more attempts
#             if normalize_finish_reason(finish_reason) == "SAFETY":
#                 print(f"  Safety block detected — skipping retries.")
#                 return answer, finish_reason
#             return answer, finish_reason
#         except Exception as e:
#             err = str(e).upper()
#             if "503" in err or "429" in err:
#                 wait_time = (attempt + 1) * 5
#                 print(f"  Attempt {attempt + 1} failed ({err[:60]}). Retrying in {wait_time}s...")
#                 time.sleep(wait_time)
#             else:
#                 return f"ERROR: {str(e)}", "EXCEPTION"

#     return "[ERROR: max retries exceeded]", "EXCEPTION"


# def main():
#     load_dotenv()
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         raise ValueError("GEMINI_API_KEY not found in .env file")

#     client = genai.Client(api_key=api_key)

#     with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
#         questions = [
#             line.strip().split(":", 1)[1].strip()
#             if line.lower().startswith("q:")
#             else line.strip()
#             for line in f
#             if line.strip()
#         ]

#     rows = []
#     blocked_count = 0

#     for i, question in enumerate(questions, start=1):
#         print(f"Processing question {i}/{len(questions)}.")

#         answer, finish_reason = generate_with_api_retry(client, question, SYSTEM_PROMPT)
#         norm_finish_reason = normalize_finish_reason(finish_reason)

#         if norm_finish_reason == "SAFETY" or answer.startswith("[BLOCKED"):
#             blocked_count += 1
#             print(f"  Q{i} blocked by safety filter.")
#         elif norm_finish_reason in {"MAX_TOKENS", "LENGTH"} or looks_truncated(answer):
#             print("  Detected likely truncation; retrying with concise prompt...")
#             concise_answer, concise_finish_reason = generate_with_api_retry(
#                 client, question, CONCISE_SYSTEM_PROMPT
#             )
#             if (
#                 concise_answer
#                 and not concise_answer.startswith("ERROR")
#                 and not concise_answer.startswith("[BLOCKED")
#             ):
#                 if not looks_truncated(concise_answer) or looks_truncated(answer):
#                     answer = concise_answer

#         answer = answer.replace("\n", " ").strip()
#         rows.append([question, answer])

#     OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
#     with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Q", "A"])
#         writer.writerows(rows)

#     print(f"\nCSV written to {OUTPUT_FILE}")
#     print(f"Total questions processed: {len(rows)}")
#     if blocked_count:
#         print(f"Safety blocks encountered: {blocked_count} question(s)")


# if __name__ == "__main__":
#     main()
import csv
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------- config ----------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
QUESTIONS_FILE = PROJECT_ROOT / "QA" / "Q2.txt"
OUTPUT_FILE = PROJECT_ROOT / "responses" / "gemini_Q2_BASE_6.csv"
MODEL = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 1024
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
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    client = genai.Client(api_key=api_key)

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

        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=build_prompt(question),
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                    ),
                )
                break
            except Exception as e:
                err = str(e).upper()
                if "503" in err or "429" in err:
                    wait_time = (attempt + 1) * 5
                    print(f"  Attempt {attempt + 1} failed ({str(e)[:60]}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

        answer = ""
        for part in (response.candidates[0].content.parts if response.candidates else []):
            if hasattr(part, "text"):
                answer += part.text
        answer = answer.replace("\n", " ").strip()

        rows.append([question, answer])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Q", "A"])
        writer.writerows(rows)

    print(f"\nCSV written to {OUTPUT_FILE}")
    print(f"Total questions processed: {len(rows)}")


if __name__ == "__main__":
    main()