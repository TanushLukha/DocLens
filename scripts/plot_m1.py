"""Draws the M1 chart for the README from docs/results/m1_tesseract.json.

    pip install -e ".[plot]"
    python scripts/plot_m1.py

Writes docs/results/m1_margins.png: Tesseract's error rate on Mozhi validation words against
the width of the border around each crop, for page modes 7 and 8.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "docs" / "results"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = {"psm7": ("Page mode 7 (one line)", "#2a78d6"), "psm8": ("Page mode 8 (one word)", "#eb6834")}


def use_plex() -> None:
    """The repository's own IBM Plex Sans, so the chart looks the same on every machine."""
    path = REPO / "fonts" / "IBMPlexSans-Regular.ttf"
    font_manager.fontManager.addfont(str(path))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()


def main() -> None:
    res = json.loads((RESULTS / "m1_tesseract.json").read_text())
    margins = res.get("margins")
    if not margins:
        sys.exit("no margins experiment in m1_tesseract.json: run scripts/eval_tesseract.py first")
    use_plex()
    rows = margins["rows"]
    xs = [r["margin"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for key, (label, color) in SERIES.items():
        ys = [100 * r[key]["cer"] for r in rows]
        ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                marker="o", markersize=8, markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=2,
                label=label, zorder=3)
        ax.annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlim(-0.04, 1.12)
    ax.set_ylim(0, 70)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_yticklabels(["0", "20%", "40%", "60%"])
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    ax.set_xlabel("Border on every side of the crop, as a share of its height", color=INK_2, fontsize=10)
    ax.set_ylabel("Character error rate", color=INK_2, fontsize=10)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=10, labelcolor=INK,
              handlelength=1.2, columnspacing=1.8, borderaxespad=0.2)
    fig.text(0.08, 0.955, "How much background a word crop needs", ha="left", va="top", fontsize=13, color=INK)
    fig.text(0.08, 0.885, f"Tesseract's best Hindi model on {margins['n']:,} Mozhi validation words",
             ha="left", va="top", fontsize=10, color=INK_2)
    fig.subplots_adjust(left=0.08, right=0.93, top=0.74, bottom=0.14)
    out = RESULTS / "m1_margins.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
