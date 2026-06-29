"""Step 00 -- check that the shared input data is in place.

EN: Verifies the Google-Drive-shared inputs sit at the paths the existing code
    expects, and FAILS (non-zero exit) on the two showstoppers:
      * missing abstract cache  -> build-corpus would silently go online
      * unexpected screening_*review*.tsv -> build-corpus reads all of them
JA: 共有データが所定パスに在るか検査する。抄録キャッシュ欠落・余分な review tsv は
    致命的とみなして異常終了する。

Reads nothing destructive. Run this first.

    python reproducibility/run_00_check_inputs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import check_inputs, print_check  # noqa: E402


def main() -> int:
    print("Checking reproduction inputs (see reproducibility/README.md for placement)...")
    check = check_inputs()
    print_check(check)
    return 0 if check.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
