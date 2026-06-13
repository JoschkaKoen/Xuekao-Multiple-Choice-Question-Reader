"""Prompt builders — one focused prompt per AI call (name, class, answers, key).

All prompts demand a single JSON object and nothing else (the thinking/streamed
calls don't enforce json_object mode, so the instruction + strip_json_fences
recover the JSON).
"""

from __future__ import annotations


def build_name_prompt(roster_names: list[str]) -> str:
    roster = "\n".join(f"  - {n}" for n in roster_names)
    return (
        "You are reading ONE field cropped from a Chinese exam answer sheet (答题卡).\n"
        "The crop shows the printed label 姓名 (name) followed by a HAND-WRITTEN student name.\n"
        "Read the hand-written name and match it to EXACTLY ONE name in the class roster below.\n\n"
        f"Roster ({len(roster_names)} names):\n{roster}\n\n"
        "Rules:\n"
        "- Return the matched name EXACTLY as written in the roster (identical characters).\n"
        "- Ignore the printed label 姓名 itself; read only the handwriting.\n"
        '- If there is no handwriting (blank) → "NONAME".\n'
        '- If there is writing but it is illegible → "UNREADABLE".\n'
        '- If you can read the handwriting but its characters do not CLEARLY correspond to a roster '
        'name, return "NOMATCH" (put your reading in raw_name). Do NOT snap to the visually nearest '
        "roster name — a wrong match corrupts that student's record, so NOMATCH for manual review is "
        "strongly preferred over guessing.\n"
        "- Two different students may share the same name; match only the characters you actually see, "
        "and do not avoid a name just because it might already be used.\n"
        "- confidence: integer 0–5 (5 = certain).\n"
        "- problem: a short note on any difficulty (ambiguous strokes, smudge, two plausible "
        "matches), or an empty string.\n\n"
        "Return ONLY this JSON, no other text:\n"
        '{"matched_name": "<roster name or NONAME/UNREADABLE/NOMATCH>", '
        '"raw_name": "<characters you read>", "confidence": <0-5>, '
        '"problem": "<short note or empty>"}'
    )


def build_class_prompt(allowed: list[str]) -> str:
    allowed_str = ", ".join(allowed)
    return (
        "You are reading ONE field cropped from a Chinese exam answer sheet (答题卡).\n"
        "The crop shows the printed label 班级 (class) followed by a HAND-WRITTEN class.\n"
        "Read it and match it to EXACTLY ONE of these allowed classes:\n"
        f"  {allowed_str}\n\n"
        "Notes:\n"
        '- Students may write the class as "EMP1".."EMP3" or "A2801".."A2804". Return whichever '
        "they wrote (it will be normalized later); prefer the EMP form if both are plausible.\n"
        "- Ignore the printed label 班级; read only the handwriting.\n"
        '- If blank → "NONE". If illegible → "UNREADABLE". If it matches none → "NOMATCH".\n'
        "- confidence: integer 0–5.\n"
        "- problem: a short note on any difficulty, or an empty string.\n\n"
        "Return ONLY this JSON:\n"
        '{"class": "<one allowed label or NONE/UNREADABLE/NOMATCH>", '
        '"class_raw": "<characters you read>", "confidence": <0-5>, '
        '"problem": "<short note or empty>"}'
    )


def build_answers_prompt(n_questions: int | None = None, double_qs: list[int] | None = None) -> str:
    n_txt = (f"There are {n_questions} questions, numbered 1..{n_questions}."
             if n_questions else
             "Read every question present on the sheet, numbered in order starting at 1.")
    dbl = ""
    if double_qs:
        dbl = ("\nSome questions are 双项选择题 (TWO correct options must be filled): "
               f"questions {', '.join(map(str, double_qs))}. For those, expect two filled options.")
    else:
        dbl = ("\nMost questions have ONE filled option, but some sheets have 双项选择题 questions "
               "with TWO filled options — always report ALL options that are filled.")
    return (
        "You are reading the multiple-choice ANSWER GRID cropped from a Chinese exam 答题卡.\n"
        "Each question lists options A B C D and the student darkens (fills) their chosen option(s).\n"
        f"{n_txt}{dbl}\n\n"
        "For EACH question, report which option letters are clearly FILLED/darkened, in order:\n"
        "- A solidly darkened box/bubble = filled. A faint, empty, or crossed-out one = NOT filled.\n"
        "- If a question has no filled option, use an empty list [].\n"
        "- If more than one option is filled, list them all.\n"
        "- Read carefully row by row; do not skip question numbers.\n\n"
        "Return ONLY this JSON (one key per question number):\n"
        '{"answers": {"1": ["B"], "2": ["A","C"], "3": []}, "confidence": <0-5>, '
        '"problem": "<short note, or empty string>"}'
    )


def build_key_prompt() -> str:
    return (
        "You are reading the OFFICIAL ANSWER KEY for the multiple-choice section of a Chinese exam "
        "(参考答案). The page(s) contain answer tables, e.g. 选择题 / 单项选择题 (single answer, "
        "often 每小题2分) and possibly 双项选择题 (TWO correct options, often 每小题3分).\n\n"
        "Extract, for EACH multiple-choice question in order starting at 1:\n"
        '- "q": the question number (integer)\n'
        '- "correct": list of correct option letters — ONE letter for single-answer questions, '
        "TWO for 双项选择题 questions\n"
        '- "points": points for that question, read from its section header (每小题X分)\n\n'
        "Ignore the 非选择题 / 主观题 / essay section entirely.\n"
        "Also report overall confidence (0–5) and a short problem note (or empty string).\n\n"
        "Return ONLY this JSON:\n"
        '{"subject": "<e.g. 历史>", "total_questions": <int>, '
        '"questions": [{"q": 1, "correct": ["C"], "points": 2}, ...], '
        '"confidence": <0-5>, "problem": "<short note or empty>"}'
    )
