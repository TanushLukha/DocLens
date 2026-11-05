import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ci_annotate import annotation, escape  # noqa: E402


def test_newlines_and_percent_signs_are_escaped():
    assert escape("50% done\nnext\r") == "50%25 done%0Anext%0D"
    assert escape("a: b, c", prop=True) == "a%3A b%2C c"


def test_only_the_end_of_the_log_is_kept():
    log = "\n".join(f"line {i}" for i in range(100))
    out = annotation(log, "Tests failed, exit code 2", lines=3)
    assert out == "::error title=Tests failed%2C exit code 2::line 97%0Aline 98%0Aline 99"
    assert annotation("", "Lint").endswith("::(the step wrote nothing)")
