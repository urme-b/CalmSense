"""Fill the AUTOGEN tables in PAPER.md from the calibration/personalization results.

Run after scripts/calibration.py and scripts/personalize.py so PAPER.md §4.7-4.8
always reflect the committed JSON, never hand-typed numbers.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "PAPER.md"
RESULTS = ROOT / "results"


def _f(x: float) -> str:
    return f"{x:.3f}"


def _replace(text: str, name: str, body: str) -> str:
    start, end = f"<!-- AUTOGEN:{name} START -->", f"<!-- AUTOGEN:{name} END -->"
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pat.search(text):
        raise SystemExit(f"AUTOGEN markers for '{name}' not found in PAPER.md")
    return pat.sub(f"{start}\n{body}\n{end}", text)


def _calibration_table() -> str:
    with open(RESULTS / "calibration.json") as fh:
        d = json.load(fh)
    head = "| Evaluation | ECE | MCE | Brier |\n| --- | :-: | :-: | :-: |"
    rows = [
        ("Within-subject 5-fold", "within_subject"),
        ("LOSO (subject-independent)", "loso"),
        ("LOSO + leak-free recalibration", "recalibrated_isotonic"),
    ]
    lines = [
        f"| {label} | {_f(d[k]['ece'])} | {_f(d[k]['mce'])} | {_f(d[k]['brier'])} |"
        for label, k in rows
    ]
    return "\n".join([head] + lines)


def _personalization_table() -> str:
    with open(RESULTS / "personalization.json") as fh:
        d = json.load(fh)
    head = "| Recalibration | ECE | Brier |\n| --- | :-: | :-: |"
    rows = [("None (LOSO)", d["uncalibrated"]), ("Global (training subjects)", d["global"])]
    # k values are whatever personalize.py used (K_VALUES), not hard-coded
    for k in sorted(d["fewshot"], key=int):
        rows.append((f"Few-shot, {k} windows", d["fewshot"][k]))
    lines = [f"| {label} | {_f(s['ece'])} | {_f(s['brier'])} |" for label, s in rows]
    return "\n".join([head] + lines)


def main() -> None:
    if (
        not (RESULTS / "calibration.json").exists()
        or not (RESULTS / "personalization.json").exists()
    ):
        sys.exit("Missing results: run scripts/calibration.py and scripts/personalize.py first.")
    text = PAPER.read_text()
    text = _replace(text, "calibration", _calibration_table())
    text = _replace(text, "personalization", _personalization_table())
    PAPER.write_text(text)
    print("PAPER.md §4.7-4.8 tables updated from results/.")


if __name__ == "__main__":
    main()
