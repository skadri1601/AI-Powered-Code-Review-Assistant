# scripts/download_pr_data.py

"""
Download PR diff–comment pairs from one or more GitHub repos
and write them out as JSONL for model training.
"""

import os
import json
from pathlib import Path

import requests
from dotenv import load_dotenv, find_dotenv
from github import Github
from unidiff import PatchSet

# ─── Configuration ───────────────────────────────────────────────────────────────

TARGET_REPOS = ["psf/requests", "pallets/flask"]
PRS_PER_REPO  = 50

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_FILE  = DATA_DIR / "pr_review_pairs.jsonl"
DATA_DIR.mkdir(exist_ok=True, parents=True)

# ─── Dotenv Loading & Debug ──────────────────────────────────────────────────────

# 1) Find the .env path — we expect it in PROJECT_ROOT
env_path = PROJECT_ROOT / ".env"
if not env_path.is_file():
    raise FileNotFoundError(f"🚨 .env file not found at expected location:\n   {env_path}")

# 2) Load it
load_dotenv(dotenv_path=env_path)

# 3) Verify
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
print(f"🔍 Loading .env from: {env_path}")
print(f"🔑 GITHUB_TOKEN loaded? {'YES' if GITHUB_TOKEN else 'NO'}")

if not GITHUB_TOKEN:
    raise RuntimeError("🚨 GITHUB_TOKEN not found in .env")

gh = Github(GITHUB_TOKEN)

# ─── Helpers ────────────────────────────────────────────────────────────────────

def extract_diff_hunks(diff_text: str) -> list[str]:
    patch = PatchSet(diff_text)
    return [str(hunk) for pf in patch for hunk in pf]

def fetch_diff(pr) -> str | None:
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.diff",
    }
    resp = requests.get(pr.diff_url, headers=headers)
    if resp.ok:
        return resp.text
    print(f"⚠️  Failed to fetch diff for PR #{pr.number}: HTTP {resp.status_code}")
    return None

# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    with OUTPUT_FILE.open("w", encoding="utf-8") as out_f:
        for repo_name in TARGET_REPOS:
            print(f"➡️  Scraping {repo_name}...")
            repo  = gh.get_repo(repo_name)
            pulls = repo.get_pulls(state="closed", sort="updated", direction="desc")
            done  = 0

            for pr in pulls:
                if done >= PRS_PER_REPO: break
                if not pr.merged:       continue

                diff_text = fetch_diff(pr)
                if not diff_text:       continue

                hunks    = extract_diff_hunks(diff_text)
                comments = pr.get_review_comments()

                for c in comments:
                    if c.position is None: continue
                    for h in hunks:
                        rec = {
                            "repo":          repo_name,
                            "pr_number":     pr.number,
                            "diff_hunk":     h,
                            "review_comment": c.body.strip(),
                        }
                        out_f.write(json.dumps(rec) + "\n")

                done += 1

            print(f"✅ Done: {done} PRs from {repo_name}")

if __name__ == "__main__":
    main()
