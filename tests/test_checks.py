import pytest

from doclens.checks import amount, fix, rules, written_like

RECEIPT = [("menu.cnt", "1"), ("menu.price", "24,000"), ("menu.sub.price", "4,000"),
           ("sub_total.subtotal_price", "SUB TOTAL 28,000"), ("total.total_price", "TOTAL 28,000"),
           ("total.cashprice", "CASH 100,000"), ("total.changeprice", "CHANGE 72,000")]


@pytest.mark.parametrize("text, value", [("TOTAL 60.000", 6000000), ("24,000", 2400000), ("12,500.00", 1250000),
                                         ("Qty 2.00", 200), ("@11000", 1100000), ("1.234.567", 123456700),
                                         ("no number", None)])
def test_amounts_read_thousands_and_cents(text, value):
    assert amount(text) == value


def test_a_value_is_written_like_the_number_it_replaces():
    assert written_like(2540000, "CHANGE 28,400") == "25,400"
    assert written_like(2540000, "28.400") == "25.400"
    assert written_like(2540050, "28,400.00") == "25,400.50"
    assert written_like(2540000, "28400") == "25400"


def test_the_rules_hold_on_a_receipt_that_adds_up():
    assert [(r.name, r.holds) for r in rules(RECEIPT)] == [
        ("items add up to the subtotal", True), ("subtotal and its extras give the total", True),
        ("cash less the total is the change", True)]


def test_tax_may_be_added_to_the_total_or_already_in_it():
    base = [("sub_total.subtotal_price", "100"), ("sub_total.tax_price", "10")]
    assert rules(base + [("total.total_price", "110")])[0].holds
    assert rules(base + [("total.total_price", "100")])[0].holds
    assert not rules(base + [("total.total_price", "105")])[0].holds


def test_one_misread_digit_is_fixed_when_only_one_field_can_take_the_blame():
    bad = list(RECEIPT)
    bad[3] = ("sub_total.subtotal_price", "SUB TOTAL 26,000")       # two rules point at the subtotal
    fixed, changes = fix(bad)
    assert fixed == RECEIPT and [c["field"] for c in changes] == ["sub_total.subtotal_price"]


def test_no_fix_when_two_fields_could_take_the_blame():
    bad = list(RECEIPT)
    bad[6] = ("total.changeprice", "CHANGE 78,000")                 # or the cash, 106,000: both one digit away
    assert fix(bad) == (bad, [])
