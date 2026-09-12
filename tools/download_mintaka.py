"""
Download the COMPLETE Mintaka dataset from the original Amazon Science release
and preserve every record field — no flattening or field selection.

Source: https://github.com/amazon-science/mintaka  (data/mintaka_{split}.json)

Why the original GitHub release and NOT the HuggingFace port:
The HF dataset (`AmazonScience/mintaka`, "en") is a LOSSY subset. It drops:
  - answer.answerType  (entity | numerical | boolean | date | string)  <- needed for
                        correct, type-aware answer scoring
  - the typed answer value (answer.answer)
  - translations (8 languages)
  - supportingEnt / supportingNum
and flattens the rich `answer` object down to answerText + answerEntity.

The original files carry the full record. JSON is parsed and serialized again,
so formatting and file hashes may differ from the download. The load-time
parsers decide which fields reach retrieval and scoring. Do not overwrite the
frozen evaluation files when reproducing the submitted experiment.

Run from repo root:
    venv\\Scripts\\python tools\\download_mintaka.py            # all splits
    venv\\Scripts\\python tools\\download_mintaka.py --splits train  # omitted split only
"""

import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path

QUESTIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "questions"
BASE_URL = "https://raw.githubusercontent.com/amazon-science/mintaka/main/data/mintaka_{split}.json"
SPLITS = ("train", "dev", "test")


def download_split(split: str) -> list[dict]:
    url = BASE_URL.format(split=split)
    print(f"Downloading COMPLETE Mintaka '{split}' split from {url}")
    with urllib.request.urlopen(url) as resp:
        raw_bytes = resp.read()
    rows = json.loads(raw_bytes.decode("utf-8"))
    print(f"  Downloaded {len(rows)} records ({len(raw_bytes)} bytes)")
    return rows


def summarize(rows: list[dict]) -> None:
    cat = Counter(r.get("category") for r in rows)
    atype = Counter((r.get("answer") or {}).get("answerType") for r in rows)
    comp = Counter(r.get("complexityType") for r in rows)
    print("    category   :", dict(sorted(cat.items())))
    print("    complexity :", dict(sorted(comp.items())))
    print("    answerType :", dict(sorted(atype.items(), key=lambda kv: -kv[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--splits", nargs="+", default=list(SPLITS), choices=SPLITS,
        help="Which splits to download (default: all three).",
    )
    args = ap.parse_args()

    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        rows = download_split(split)
        out = QUESTIONS_DIR / f"mintaka_{split}_raw.json"
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved complete records to {out}")
        summarize(rows)
        print()


if __name__ == "__main__":
    main()
