"""Exit 0 if raw data changed since the last successful training run, 99 if not.

`--commit` records the current fingerprint (run only after training succeeds).
"""
import hashlib, json, sys
from pathlib import Path

RAW_DIR = Path("/opt/airflow/data/unzipped")
STATE = Path("/opt/airflow/data/.last_trained_fingerprint.json")
FILES = ["train_transaction.csv", "train_identity.csv",
         "test_transaction.csv", "test_identity.csv"]
SKIP_CODE = 99


def fingerprint() -> dict:
    out = {}
    for name in FILES:
        p = RAW_DIR / name
        if not p.exists():
            sys.exit(f"missing raw file: {p}")   # non-zero, non-99 -> task FAILS (good)
        st = p.stat()
        # size + mtime is cheap; add a hash of the first/last 1 MB to catch same-size edits
        h = hashlib.sha256()
        with p.open("rb") as f:
            h.update(f.read(1 << 20))
            f.seek(max(st.st_size - (1 << 20), 0))
            h.update(f.read())
        out[name] = {"size": st.st_size, "mtime": int(st.st_mtime), "sha": h.hexdigest()}
    return out


def main() -> int:
    current = fingerprint()
    if "--commit" in sys.argv:
        STATE.write_text(json.dumps(current, indent=2))
        print("fingerprint saved")
        return 0
    previous = json.loads(STATE.read_text()) if STATE.exists() else None
    if previous == current:
        print("no new data since last successful run; skipping")
        return SKIP_CODE
    print("new data detected" if previous else "first run; no prior fingerprint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
