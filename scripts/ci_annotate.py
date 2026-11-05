"""Turns the end of a failed CI step's output into a GitHub annotation.

    python scripts/ci_annotate.py <log file> <title>

GitHub shows annotations on a run's page and returns them from its public API (a check run's
annotations), which needs no sign in, unlike the full logs. So the reason a step failed can be read
from anywhere, by a person or a script. The last lines of the log become one error annotation.
"""
import sys
from pathlib import Path

LINES = 60


def escape(text: str, prop: bool = False) -> str:
    """GitHub's escaping for a workflow command: always %, CR and LF; in a property also : and ,"""
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return text.replace(":", "%3A").replace(",", "%2C") if prop else text


def annotation(log: str, title: str, lines: int = LINES) -> str:
    tail = "\n".join(log.splitlines()[-lines:]) or "(the step wrote nothing)"
    return f"::error title={escape(title, prop=True)}::{escape(tail)}"


def main() -> None:
    path, title = Path(sys.argv[1]), sys.argv[2]
    log = path.read_text(encoding="utf8", errors="replace") if path.exists() else ""
    print(annotation(log, title))


if __name__ == "__main__":
    main()
