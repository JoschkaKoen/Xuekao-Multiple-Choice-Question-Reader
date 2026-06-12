"""Grading — exact set-match against the key, weights from the key.

Covers single-answer (_1) and double-answer 双项 (_2) uniformly: a question
scores its points iff the student's filled-option SET equals the key's set;
blank / over-select / under-select / wrong all score 0 (per 不选/多选/错选均不得分).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def norm_set(letters) -> frozenset:
    out = set()
    if isinstance(letters, str):
        letters = [letters]
    for x in (letters or []):
        for c in str(x).upper():
            if c in "ABCD":
                out.add(c)
    return frozenset(out)


def grade_question(marked: frozenset, correct: frozenset, points: float):
    if not marked:
        return 0.0, "blank"
    if marked == correct:
        return points, "correct"
    if marked < correct:
        return 0.0, "under"
    if marked > correct:
        return 0.0, "over"
    return 0.0, "wrong"


@dataclass
class StudentScore:
    per_q: dict = field(default_factory=dict)   # q -> (marked_set, tag, points)
    total: float = 0.0
    n_correct: int = 0
    read_n: int = 0
    key_n: int = 0
    problems: list = field(default_factory=list)


def grade_student(answers: dict, key: dict) -> StudentScore:
    per_q: dict = {}
    total = 0.0
    n_correct = 0
    for it in key["questions"]:
        q = it["q"]
        correct = norm_set(it["correct"])
        marked = norm_set(answers.get(str(q), answers.get(q, [])))
        pts, tag = grade_question(marked, correct, it["points"])
        per_q[q] = (marked, tag, pts)
        total += pts
        n_correct += tag == "correct"
    read_qs = {int(k) for k in answers.keys() if str(k).isdigit()}
    key_set = {it["q"] for it in key["questions"]}
    problems = []
    missing = key_set - read_qs
    extra = read_qs - key_set
    if missing:
        problems.append("missing Qs: " + ",".join(map(str, sorted(missing))))
    if extra:
        problems.append("extra Qs: " + ",".join(map(str, sorted(extra))))
    return StudentScore(per_q, round(total, 2), n_correct,
                        len(read_qs & key_set), len(key_set), problems)


def item_stats(scores: list, key: dict) -> list[dict]:
    """Per-question stats across all graded students (the per-exercise table)."""
    graded = [s for s in scores if s is not None]
    n = len(graded)
    stats = []
    for it in key["questions"]:
        q = it["q"]
        n_correct = blank = answered = 0
        for s in graded:
            _m, tag, _p = s.per_q.get(q, (frozenset(), "blank", 0.0))
            if tag == "correct":
                n_correct += 1
            if tag == "blank":
                blank += 1
            else:
                answered += 1
        stats.append({
            "q": q, "correct": "".join(it["correct"]), "points": it["points"],
            "answered": answered, "n_correct": n_correct,
            "pct": round(n_correct / n * 100, 1) if n else 0.0,
            "blank": blank, "wrong": answered - n_correct,
        })
    return stats
