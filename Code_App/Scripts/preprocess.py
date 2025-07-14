# scripts/preprocess.py

"""
Read data/pr_review_pairs.jsonl, clean and split into train/val/test,
and write out JSONL files suitable for Hugging Face’s Trainer API.
"""

import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# ─── Config ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
RAW_FILE      = PROJECT_ROOT / "data" / "pr_review_pairs.jsonl"
OUT_DIR       = PROJECT_ROOT / "data" / "hf_dataset"
TEST_SIZE     = 0.1
VAL_SIZE      = 0.1  # of remaining after test split
RANDOM_SEED   = 42

# Make output dir
OUT_DIR.mkdir(exist_ok=True)

def load_pairs(path):
    """Yield dicts with keys 'input' and 'target'."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            yield {
                "input":  rec["diff_hunk"],
                "target": rec["review_comment"]
            }

def save_jsonl(records, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    # 1) Load and shuffle
    all_records = list(load_pairs(RAW_FILE))
    random.seed(RANDOM_SEED)
    random.shuffle(all_records)

    # 2) Split out test set
    test_count = int(len(all_records) * TEST_SIZE)
    test_set   = all_records[:test_count]
    rest       = all_records[test_count:]

    # 3) Split rest into train/val
    val_count  = int(len(rest) * VAL_SIZE)
    val_set    = rest[:val_count]
    train_set  = rest[val_count:]

    # 4) Save to files
    save_jsonl(train_set, OUT_DIR / "train.jsonl")
    save_jsonl(val_set,   OUT_DIR / "validation.jsonl")
    save_jsonl(test_set,  OUT_DIR / "test.jsonl")

    print(f"💾 Saved {len(train_set)} train, {len(val_set)} val, {len(test_set)} test records")
    print(f"→ Files in {OUT_DIR}")

if __name__ == "__main__":
    main()
