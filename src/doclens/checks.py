"""Arithmetic checks on a receipt's fields, and fixes of one character (M6).

A receipt states its own sums. On CORD's receipts three rules can be checked from the fields:
the items (their prices and add ons, less item discounts) add up to the subtotal, or to the total
when there is no subtotal; the subtotal with service, less discounts and with or without tax
(receipts print tax both ways) gives the total; and cash less the total is the change. M4 found
that most fields the OCR loses are one character off, often one digit. When a rule fails and one
field can be changed by exactly one character so that the rule holds, while no other field of the
rule can, that change is made: the sum then says which digit was misread.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

NUMBER = re.compile(r"\d(?:[\d.,]*\d)?")
ITEMS, ADD_ONS, ITEM_DISCOUNTS = "menu.price", "menu.sub.price", "menu.discountprice"
SUBTOTAL, TAX, SERVICE, DISCOUNT = ("sub_total.subtotal_price", "sub_total.tax_price", "sub_total.service_price",
                                    "sub_total.discount_price")
TOTAL, CASH, CHANGE = "total.total_price", "total.cashprice", "total.changeprice"


def amount(text: str) -> int | None:
    """The last number in a field's text, in hundredths: separators followed by three digits group
    thousands, a final separator followed by two digits marks the cents ("60.000", "24,000" and
    "24,000.00" are all 2,400,000 hundredths). None when the text holds no number."""
    found = NUMBER.findall(text)
    if not found:
        return None
    n = found[-1]
    m = re.fullmatch(r"(.*\d)[.,](\d{2})", n)
    whole, cents = (m.group(1), m.group(2)) if m and not re.search(r"[.,]\d{3}$", n) else (n, "00")
    digits = re.sub(r"[.,]", "", whole)
    return int(digits) * 100 + int(cents)


def written_like(value: int, model: str) -> str:
    """A value in hundredths written as the number in `model` is: its thousands separator, its cents."""
    n = NUMBER.findall(model)[-1]
    seps = [c for c in n if c in ".,"]
    cents = bool(re.search(r"[.,]\d{2}$", n)) and not re.search(r"[.,]\d{3}$", n)
    group = (seps[0] if seps and (not cents or len(seps) > 1) else "")
    whole = str(abs(value) // 100)
    if group:
        whole = f"{int(whole):,}".replace(",", group)
    text = whole + (f"{seps[-1]}{abs(value) % 100:02d}" if cents else "")
    return ("-" if value < 0 else "") + text


def _replace_number(text: str, new: str) -> str:
    last = list(NUMBER.finditer(text))[-1]
    return text[:last.start()] + new + text[last.end():]


@dataclass
class Rule:
    name: str
    holds: bool
    left: int
    right: int


def _values(fields: list[tuple[str, str]]) -> dict[str, list[tuple[int, int]]]:
    """Per category, (field index, amount) of every field with a number."""
    out: dict[str, list[tuple[int, int]]] = {}
    for i, (cat, text) in enumerate(fields):
        a = amount(text)
        if a is not None:
            out.setdefault(cat, []).append((i, a))
    return out


def _equations(v: dict[str, list[tuple[int, int]]]) -> list[tuple[str, list[tuple[int, int]], list[int]]]:
    """Every rule that applies: (name, terms, accepted totals), where a term is (field index, sign)
    and the rule holds when the signed sum of its terms is one of the accepted values (zero, or
    zero once tax is added, for receipts that print the total without tax)."""
    def one(cat):
        return v[cat][0] if len(v.get(cat, [])) == 1 else None

    out = []
    items = [(i, 1) for i, _ in v.get(ITEMS, [])] + [(i, 1) for i, _ in v.get(ADD_ONS, [])]
    items += [(i, -1 if a > 0 else 1) for i, a in v.get(ITEM_DISCOUNTS, [])]
    target = one(SUBTOTAL) or one(TOTAL)
    if v.get(ITEMS) and target:
        out.append(("items add up to the " + ("subtotal" if one(SUBTOTAL) else "total"),
                    items + [(target[0], -1)], [0]))
    sub, tot = one(SUBTOTAL), one(TOTAL)
    if sub and tot:
        terms = [(sub[0], 1), (tot[0], -1)] + [(i, 1) for i, _ in v.get(SERVICE, [])]
        terms += [(i, -1 if a > 0 else 1) for i, a in v.get(DISCOUNT, [])]
        tax = one(TAX)
        out.append(("subtotal and its extras give the total", terms + ([(tax[0], 1)] if tax else []),
                    [0] + ([tax[1]] if tax else [])))               # tax added to the total, or already in it
    cash, change = one(CASH), one(CHANGE)
    if cash and tot and change:
        out.append(("cash less the total is the change", [(cash[0], 1), (tot[0], -1), (change[0], -1)], [0]))
    return out


def rules(fields: list[tuple[str, str]]) -> list[Rule]:
    v = _values(fields)
    amounts = {i: a for pairs in v.values() for i, a in pairs}
    out = []
    for name, terms, accepted in _equations(v):
        s = sum(sign * amounts[i] for i, sign in terms)
        out.append(Rule(name, s in accepted, s, accepted[0]))
    return out


def fix(fields: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[dict]]:
    """Fields with one character changes that make failing rules hold (see the module's text), and
    what was changed. A change is a candidate when it makes its rule hold and breaks no rule that
    held; it is made only when it is the rule's one candidate."""
    fields = list(fields)
    changes: list[dict] = []
    for _ in range(3):                                   # a fix can let a later rule be checked
        before = rules(fields)
        held = {r.name for r in before if r.holds}
        v = _values(fields)
        amounts = {i: a for pairs in v.values() for i, a in pairs}
        made = False
        for name, terms, accepted in _equations(v):
            s = sum(sign * amounts[i] for i, sign in terms)
            if s in accepted:
                continue
            options = set()
            for i, sign in terms:
                for goal in accepted:
                    need = amounts[i] - sign * (s - goal)          # the amount that makes the sum reach goal
                    if need < 0:
                        continue
                    old = NUMBER.findall(fields[i][1])[-1]
                    new = written_like(need, fields[i][1])
                    if Levenshtein.distance(old, new) != 1:
                        continue
                    trial = list(fields)
                    trial[i] = (fields[i][0], _replace_number(fields[i][1], new))
                    after = {r.name for r in rules(trial) if r.holds}
                    if name in after and held <= after:
                        options.add((i, trial[i][1]))
            if len(options) == 1:
                ((i, text),) = options
                changes.append({"rule": name, "field": fields[i][0], "from": fields[i][1], "to": text})
                fields[i] = (fields[i][0], text)
                made = True
                break
        if not made:
            break
    return fields, changes
