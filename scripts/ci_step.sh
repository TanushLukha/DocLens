#!/usr/bin/env bash
# Runs one CI step and, when it fails, turns the end of its output into a GitHub annotation
# (scripts/ci_annotate.py), so the reason can be read through GitHub's public API.
#   bash scripts/ci_step.sh "Tests" pytest -q -rfE
title="$1"
shift
log="$(mktemp)"
"$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  python3 "$(dirname "$0")/ci_annotate.py" "$log" "$title failed with exit code $status"
fi
exit "$status"
