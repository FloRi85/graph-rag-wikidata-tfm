"""
Publication-quality figures describing the Mintaka benchmark.

Reads selected raw splits (github.com/amazon-science/mintaka) and renders three
figures for thesis section 4.2 "Dataset: Mintaka", plus a text summary of the data
quality facts that are easier to state than to draw. The thesis figures default
to the complete 4,000-question test split; pass ``--splits train dev test`` to
describe all 20,000 English questions instead.

  Fig 1  mintaka_overview            answer types, entity anchors, gold cardinality,
                                     question length
  Fig 2  answer_type_by_complexity   the headline: answerType is NOT complexityType
  Fig 3  sample_representativeness    does the stratified dev-200 track the population?

Axis labels use the QUALIFIED vocabularies from src/eval/parse_questions.py and
parse_answers.py (c_ / q_ /
a_ prefixes), because `entity`, `ordinal` and `date` each name BOTH a questionEntity
type and an answer type - on a chart with both axes in play the bare words are
ambiguous. The prefix is display-only; the underlying data keeps the raw Mintaka value.

Colour is the validated reference palette, not a matplotlib default. The categorical
slots were checked with the data-viz validator in both themes (all-pairs: light CVD
dE 9.2 / normal-vision 24.0; dark 9.4 / 20.9, ALL CHECKS PASS). Slot 3 (aqua) measures
2.74:1 on the light surface, below the 3:1 bar, so every series carrying it ships
visible direct labels - the palette's "relief rule".

Run from repo root:
    venv\\Scripts\\python tools\\plot_mintaka_stats.py
    venv\\Scripts\\python tools\\plot_mintaka_stats.py --splits train dev test
    venv\\Scripts\\python tools\\plot_mintaka_stats.py --out output/figures
    venv\\Scripts\\python tools\\plot_mintaka_stats.py --theme dark --formats png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless: never try to open a window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.parse_questions import (  # noqa: E402
    COMPLEXITY_TYPES, raw_type,
)
from src.eval.parse_answers import ANSWER_TYPES, load_answers  # noqa: E402

QDIR = ROOT / "data" / "questions"
SPLITS = ("train", "dev", "test")

# ── Theme ────────────────────────────────────────────────────────────────────
# Both themes are SELECTED, not flipped: the dark series are the palette's own dark
# steps of the same three hues, validated against the dark surface.
THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        series=("#2a78d6", "#eb6834", "#1baf7a"),
        seq=("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"),
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        series=("#3987e5", "#d95926", "#199e70"),
        seq=("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"),
    ),
}


def _apply_theme(t: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["ink"],
        "axes.labelcolor": t["ink2"],
        "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "figure.dpi": 110,
    })


def _style(ax, t: dict, *, grid_axis: str = "y") -> None:
    """Recessive chrome: no top/right spines, hairline grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(0.8)
    if grid_axis != "none":
        ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ── Data ─────────────────────────────────────────────────────────────────────
def load_raw(splits) -> pd.DataFrame:
    """One row per question, with the fields the flattened schema does not keep."""
    rows = []
    for split in splits:
        path = QDIR / f"mintaka_{split}_raw.json"
        if not path.exists():
            sys.exit(f"missing {path} - run tools/download_mintaka.py first")
        for r in json.loads(path.read_text(encoding="utf-8")):
            ents = [m for m in (r.get("questionEntity") or [])
                    if m.get("entityType") == "entity" and m.get("name")]
            ans = r.get("answer") or {}
            payload = ans.get("answer") or []
            rows.append({
                "id": r["id"],
                "split": split,
                "category": r.get("category", ""),
                "complexity": r.get("complexityType", ""),
                "answer_type": ans.get("answerType", ""),
                "n_qids": len(set(m["name"] for m in ents)),
                "n_gold": len(payload) if isinstance(payload, list) else 1,
                "words": len(r["question"].split()),
            })
    return pd.DataFrame(rows)


def _sample_share(path: Path) -> pd.Series | None:
    """answer_type share (%) of a sample file, or None when the file is absent.

    🔴 RETURNS None, NOT AN EMPTY SERIES, AND THE DIFFERENCE IS A WRONG FIGURE.
    An empty Series survives `.reindex(order).fillna(0)` as a row of zeros, so
    a deleted sample file used to render as a full series of 0.0% bars — legend
    entry, direct labels and all — with nothing to say the data was missing.
    Caught 2026-08-16 after `mintaka_sample_100.json` was removed. "Not measured"
    and "measured as zero" must never share a representation in a thesis figure.
    """
    if not path.exists():
        return None
    golds = load_answers(path)
    return (pd.Series([g.answer_type for g in golds.values()])
            .value_counts(normalize=True) * 100)


def _population_label(df: pd.DataFrame) -> str:
    """Human-readable provenance for the rows underlying a figure."""
    present = set(df.split)
    splits = tuple(split for split in SPLITS if split in present)
    if splits == SPLITS:
        return "Full Mintaka"
    if len(splits) == 1:
        return f"Mintaka {splits[0].upper()} split"
    return "Mintaka " + "+".join(split.upper() for split in splits) + " splits"


# ── Fig 1 ────────────────────────────────────────────────────────────────────
def fig_overview(df: pd.DataFrame, t: dict):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    fig.suptitle(f"{_population_label(df)} composition  ·  n = {len(df):,} questions",
                 fontsize=13, fontweight="bold", color=t["ink"], y=0.98)
    blue = t["series"][0]

    # (0,0) answer type - horizontal bars, direct labels
    ax = axes[0, 0]
    order = [raw_type(a) for a in ANSWER_TYPES]
    counts = df.answer_type.value_counts().reindex(order).fillna(0)
    y = np.arange(len(order))
    ax.barh(y, counts.values, color=blue, height=0.62, zorder=3)
    ax.set_yticks(y, ANSWER_TYPES, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.22)
    for i, v in enumerate(counts.values):
        ax.text(v + counts.max() * 0.02, i, f"{int(v):,}  ({v / len(df) * 100:.1f}%)",
                va="center", fontsize=8, color=t["ink2"])
    ax.set_title("Answer type", loc="left", color=t["ink"])
    ax.set_xlabel("questions", fontsize=8)
    _style(ax, t, grid_axis="x")

    # (0,1) distinct QIDs per question
    ax = axes[0, 1]
    nq = df.n_qids.clip(upper=4).value_counts().sort_index()
    labels = [str(i) for i in nq.index[:-1]] + ["4+"] if nq.index.max() == 4 else [str(i) for i in nq.index]
    ax.bar(np.arange(len(nq)), nq.values, color=blue, width=0.62, zorder=3)
    ax.set_xticks(np.arange(len(nq)), labels)
    for i, v in enumerate(nq.values):
        ax.text(i, v + nq.max() * 0.02, f"{int(v):,}", ha="center", fontsize=8, color=t["ink2"])
    ax.set_ylim(0, nq.max() * 1.16)
    zero = int((df.n_qids == 0).sum())
    ax.set_title("Distinct Wikidata anchors per question", loc="left", color=t["ink"])
    ax.set_xlabel(f"QIDs  ·  {zero} questions have none (Graph-RAG gets empty context)", fontsize=8)
    _style(ax, t)

    # (1,0) gold answer cardinality
    ax = axes[1, 0]
    ng = df.n_gold.clip(upper=3).value_counts().sort_index()
    lab = {0: "0", 1: "1", 2: "2", 3: "3+"}
    ax.bar(np.arange(len(ng)), ng.values, color=blue, width=0.62, zorder=3)
    ax.set_xticks(np.arange(len(ng)), [lab.get(i, str(i)) for i in ng.index])
    for i, v in enumerate(ng.values):
        ax.text(i, v + ng.max() * 0.02, f"{int(v):,}", ha="center", fontsize=8, color=t["ink2"])
    ax.set_ylim(0, ng.max() * 1.16)
    multi = int((df.n_gold > 1).sum())
    ax.set_title("Gold answers per question", loc="left", color=t["ink"])
    ax.set_xlabel(f"{multi:,} questions ({multi / len(df) * 100:.1f}%) are multi-answer "
                  f"-> set precision/recall", fontsize=8)
    _style(ax, t)

    # (1,1) question length
    ax = axes[1, 1]
    ax.hist(df.words, bins=np.arange(2.5, df.words.max() + 1.5, 1),
            color=blue, zorder=3)
    med = df.words.median()
    ax.axvline(med, color=t["series"][1], linewidth=2, zorder=4)
    ax.text(med + 0.6, ax.get_ylim()[1] * 0.9, f"median {med:.0f}",
            fontsize=8, color=t["series"][1], fontweight="bold")
    ax.set_title("Question length", loc="left", color=t["ink"])
    ax.set_xlabel(f"words  ·  mean {df.words.mean():.1f}, max {df.words.max()}", fontsize=8)
    ax.set_xlim(0, 30)
    _style(ax, t)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


# ── Fig 2 ────────────────────────────────────────────────────────────────────
def fig_type_by_complexity(df: pd.DataFrame, t: dict):
    rows = [raw_type(c) for c in COMPLEXITY_TYPES]
    cols = [raw_type(a) for a in ANSWER_TYPES]
    ct = pd.crosstab(df.complexity, df.answer_type).reindex(index=rows, columns=cols).fillna(0)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100

    cmap = LinearSegmentedColormap.from_list("seq_blue", list(t["seq"]))
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    im = ax.imshow(pct.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(len(cols)), ANSWER_TYPES, fontsize=9)
    ax.set_yticks(np.arange(len(rows)), COMPLEXITY_TYPES, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=t["surface"], linewidth=2.5)
    ax.tick_params(which="minor", length=0)

    for i in range(len(rows)):
        for j in range(len(cols)):
            share, n = pct.iat[i, j], int(ct.iat[i, j])
            if n == 0:
                ax.text(j, i, "·", ha="center", va="center", color=t["muted"], fontsize=11)
                continue
            # "0%" would be a lie for a cell holding real questions. Every
            # non-zero share below 1% therefore prints as "<1%"; this also avoids
            # Python's ties-to-even formatting turning an exact 0.5% into 0%.
            label = f"{share:.0f}%" if share >= 1 else "<1%"
            # Contrast is a property of the CELL FILL, not the page theme: the ramp is
            # the same in both themes, so a pale cell needs dark ink even in dark mode.
            # (Using the theme's ink here put white text on pale blue and lost the
            # small-count cells entirely.)
            ax.text(j, i, f"{label}\n{n:,}", ha="center", va="center", fontsize=7.6,
                    color="#ffffff" if share > 55 else "#0b0b0b",
                    fontweight="bold" if share > 90 else "normal")

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("share of the complexity type's questions (%)", fontsize=8, color=t["ink2"])
    cb.ax.tick_params(labelsize=8, length=0, colors=t["muted"])
    cb.outline.set_visible(False)

    ax.set_title("Answer type is not complexity type", loc="left",
                 fontsize=12.5, fontweight="bold", color=t["ink"], pad=12)
    # Axes-relative, BELOW the plot: in data coords a negative row sits above the
    # top cell and lands on top of the title.
    ax.text(0, -0.085, "Each row sums to 100%. The scorer dispatches its matcher on the "
                       "column, never the row — a `count` question is always numerical, "
                       "but a `comparative` one\nmay be boolean, entity or string.",
            transform=ax.transAxes, va="top", fontsize=8.4, color=t["ink2"])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# ── Fig 3 ────────────────────────────────────────────────────────────────────
def fig_representativeness(df: pd.DataFrame, t: dict):
    order = [raw_type(a) for a in ANSWER_TYPES]
    pop = (df.answer_type.value_counts(normalize=True) * 100).reindex(order).fillna(0)

    # ⚠️ A SAMPLE THAT IS NOT ON DISK IS OMITTED, NOT DRAWN AS ZERO. Its absence
    # is stated in the caption and on stdout, so a reader of the figure and a
    # reader of the console both learn the same thing.
    wanted = [("DEV-200 sample", QDIR / "mintaka_sample_dev_200.json")]
    population = _population_label(df)
    series = [(f"{population} (n={len(df):,})", pop, t["series"][0])]
    shares: dict[str, pd.Series] = {}
    absent: list[str] = []
    for label, path in wanted:
        share = _sample_share(path)
        if share is None:
            absent.append(f"{label} ({path.name})")
            continue
        shares[label] = share.reindex(order).fillna(0)
        series.append((label, shares[label], t["series"][len(series)]))

    if absent:
        # ASCII: this tool does not reconfigure stdout, and the Windows console
        # is cp1252 -- a warning that raises UnicodeEncodeError is not a warning.
        print(f"  WARNING representativeness: omitting {', '.join(absent)} -- file "
              f"not found. The figure shows only the samples that exist.")

    x = np.arange(len(order))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    for k, (label, vals, colour) in enumerate(series):
        # Centre whatever number of series survived, rather than assuming three.
        pos = x + (k - (len(series) - 1) / 2) * w
        ax.bar(pos, vals.values, width=w - 0.03, label=label, color=colour, zorder=3)
        # Direct labels on every bar: slot 3 is sub-3:1 on the light surface, so the
        # palette's relief rule requires them rather than leaving colour to carry it.
        for xi, v in zip(pos, vals.values):
            ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", fontsize=7.4, color=t["ink2"])

    ax.set_xticks(x, ANSWER_TYPES, fontsize=9.5)
    ax.set_ylabel("share of questions (%)", fontsize=9)
    ax.set_ylim(0, max(v.max() for _, v, _ in series) * 1.20)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=t["ink2"])
    ax.set_title(f"Do the evaluation samples track the {population}?", loc="left",
                 fontsize=12.5, fontweight="bold", color=t["ink"], pad=10)

    notes = []
    if "DEV-200 sample" in shares:
        dev = shares["DEV-200 sample"]
        notes.append(f"DEV-200 tracks the {population} closely (entity "
                     f"{dev.get('entity', 0):.1f}% vs {pop.get('entity', 0):.1f}%).")
    if absent:
        notes.append(f"NOT SHOWN: {', '.join(absent)} - file not present.")
    ax.text(0.0, -0.155, " ".join(notes), transform=ax.transAxes,
            fontsize=8.4, color=t["ink2"], va="top")
    _style(ax, t)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    return fig


# ── Text summary ─────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    print("=" * 74)
    print(f"MINTAKA  n = {len(df):,}   " +
          "  ".join(f"{s}={int((df.split == s).sum()):,}" for s in df.split.unique()))
    print("=" * 74)

    print("\nanswer_type")
    for a in ANSWER_TYPES:
        n = int((df.answer_type == raw_type(a)).sum())
        print(f"  {a:<12} {n:>7,}  {n / len(df) * 100:>5.1f}%")

    # A perfectly uniform grid is a sentence, not a chart.
    grid = pd.crosstab(df.category, df.complexity)
    vals = sorted(set(grid.values.ravel().tolist()))
    print(f"\ncategory x complexity: {grid.shape[0]} x {grid.shape[1]} grid, "
          f"cell values {vals} -> balanced by construction (not plotted)")

    print("\nDATA QUALITY FLAGS")
    zero_q = df[df.n_qids == 0]
    zero_a = df[df.n_gold == 0]
    print(f"  {len(zero_q):>5,} questions have NO Wikidata anchor "
          f"-> Config 3/4 receive an empty context")
    print(f"  {len(zero_a):>5,} questions have an EMPTY gold answer payload")
    print(f"  {int((df.n_gold > 1).sum()):>5,} questions are multi-answer "
          f"-> scored with set precision/recall")

    odd = df[(df.complexity == "yesno") & (df.answer_type != "boolean")]
    if len(odd):
        print(f"  {len(odd):>5,} 'yesno' question(s) carry a non-boolean answer "
              f"(id: {', '.join(odd.id.head(3))})")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Render Mintaka dataset figures.")
    ap.add_argument("--splits", nargs="+", default=["test"], choices=list(SPLITS))
    ap.add_argument("--out", default=str(ROOT / "data" / "analysis" / "figures"))
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    ap.add_argument("--theme", default="light", choices=list(THEMES))
    args = ap.parse_args()

    theme = THEMES[args.theme]
    _apply_theme(theme)

    df = load_raw(args.splits)
    print_summary(df)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "mintaka_overview": fig_overview(df, theme),
        "answer_type_by_complexity": fig_type_by_complexity(df, theme),
        "sample_representativeness": fig_representativeness(df, theme),
    }
    suffix = "" if args.theme == "light" else f"_{args.theme}"

    print(f"\nWriting to {out}")
    for name, fig in figures.items():
        for ext in args.formats:
            path = out / f"{name}{suffix}.{ext}"
            fig.savefig(path, bbox_inches="tight", dpi=200 if ext == "png" else None)
            print(f"  {path.name}")
        plt.close(fig)


if __name__ == "__main__":
    main()
