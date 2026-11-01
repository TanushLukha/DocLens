"""M4: which fields the OCR words lose before any model tags them, and how.

    python scripts/ocr_losses.py            results to docs/results/m4_ocr_losses.md and .json

The OCR words of the test receipts (scripts/prepare_m4.py) are given perfect tags: each takes the
field of the true words under it. A true field these tags cannot give is lost to the OCR itself.
Each lost field is set beside the closest field of its category that the tags do give, and counted
by how many characters apart they are (spaces left out, as in field F1).
"""
import json
import sys
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doclens import REPO  # noqa: E402
from doclens.fields import cord_receipts, decode, ocr_receipts, transfer  # noqa: E402

KINDS = ("one character wrong", "two or three characters wrong", "more characters wrong, or joined or split",
         "no field of its category found")


def key(field: tuple[str, str]) -> tuple[str, str]:
    return field[0], "".join(field[1].split())


def main():
    truths = cord_receipts("test", image="none")
    ocr = ocr_receipts("test", truths)
    kinds: Counter = Counter()
    by_category: Counter = Counter()
    total = numbers = numbers_lost = digits = 0
    examples = []
    for t, o in zip(truths, ocr, strict=True):
        moved = transfer(o, t)
        found = [key(f) for f in decode(moved.words, moved.tags)]
        left = Counter(found)
        for f in t.fields():
            c, text = key(f)
            total += 1
            number = any(ch.isdigit() for ch in text) and not any(ch.isalpha() for ch in text)
            numbers += number
            if left[(c, text)] > 0:
                left[(c, text)] -= 1
                continue
            numbers_lost += number
            by_category[c] += 1
            same = [x for k, x in found if k == c]
            if not same:
                kinds[KINDS[3]] += 1
                continue
            near = min(same, key=lambda x: Levenshtein.distance(text, x))
            d = Levenshtein.distance(text, near)
            kinds[KINDS[0] if d == 1 else KINDS[1] if d <= 3 else KINDS[2]] += 1
            if d == 1:
                op = Levenshtein.editops(text, near)[0]
                chars = (text[op.src_pos] if op.tag != "insert" else "") + (near[op.dest_pos] if op.tag != "delete" else "")
                digits += all(ch.isdigit() or ch in ".,:" for ch in chars)
                if len(examples) < 10:
                    examples.append({"category": c, "true": text, "read": near})
    lost = sum(kinds.values())
    res = {"receipts": len(truths), "fields": total, "lost": lost, "kinds": {k: kinds[k] for k in KINDS},
           "one_character_a_digit_or_separator": digits,
           "numbers": numbers, "numbers_lost": numbers_lost, "by_category": dict(by_category.most_common()),
           "examples": examples}
    out = REPO / "docs" / "results"
    (out / "m4_ocr_losses.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf8")
    lines = ["# M4: fields the OCR words lose\n",
             f"CORD's {len(truths)} test receipts, {total} true fields. Given perfect tags (the field of the true words"
             f" under each OCR word), the OCR words still lose {lost} fields ({100 * lost / total:.1f}%): numbers"
             f" {numbers_lost} of {numbers} ({100 * numbers_lost / max(numbers, 1):.1f}%), other fields {lost - numbers_lost}"
             f" of {total - numbers} ({100 * (lost - numbers_lost) / max(total - numbers, 1):.1f}%).\n",
             "| How the closest field of the same category differs | Fields | Share |", "|---|---:|---:|"]
    lines += [f"| {k} | {kinds[k]} | {100 * kinds[k] / max(lost, 1):.0f}% |" for k in KINDS]
    lines += ["", f"Of the {kinds[KINDS[0]]} fields one character off, {digits} differ in a digit or a separator (. , :).",
              "", "| Category | Fields lost |", "|---|---:|"] + [f"| {c} | {n} |" for c, n in by_category.most_common()]
    lines += ["", "One character apart (true, read): " + "; ".join(f"{e['true']} as {e['read']}" for e in examples)]
    (out / "m4_ocr_losses.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"{lost} of {total} fields lost to the OCR; {dict(kinds)}; wrote {out / 'm4_ocr_losses.md'} and .json")


if __name__ == "__main__":
    main()
