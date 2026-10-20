"""Run the full reproduction pipeline (steps 00 -> 08), then verify.

EN: OS-agnostic orchestrator. Each step is the corresponding run_0N script. Stops
    at the first failure. The secondary G-test suite (step 05) is skipped by
    default (pass --with-frequency to include it). Includes sensitivity analyses
    and the speaker-label tally, with a corpus-based sample check when inputs exist.
JA: OS 非依存のオーケストレータ。各ステップは run_0N スクリプトを呼ぶ。最初の失敗で停止。
    副次の G²（step 05）は既定でスキップ（--with-frequency で実行）。感度分析と話者ラベル集計、
    入力がある場合はコーパスからの標本再抽出チェックも行う。

    python reproducibility/run_all.py
    python reproducibility/run_all.py --with-frequency --force-detect
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _env import subprocess_env  # noqa: E402


def step(script: str, *args: str) -> None:
    cmd = [sys.executable, str(HERE / script), *args]
    print(f"\n########## {script} {' '.join(args)} ##########", flush=True)
    r = subprocess.run(cmd, env=subprocess_env())
    if r.returncode != 0:
        raise SystemExit(f"[ABORT] {script} failed with exit {r.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-frequency", action="store_true",
                    help="also run the optional secondary G-test suite (step 05)")
    ap.add_argument("--force-detect", action="store_true",
                    help="pass --force to step 03 (delete existing detection outputs)")
    args = ap.parse_args()

    step("run_00_check_inputs.py")
    step("run_01_build_corpus.py")
    step("run_02_strip_references.py")
    step("run_03_detect_deontic.py", *(["--force"] if args.force_detect else []))
    step("run_04_meta_analysis.py")
    if args.with_frequency:
        step("run_05_frequency_suite.py")
    step("run_06_table1.py")
    step("run_07_sensitivity.py")
    step("run_08_speaker_check.py")
    step("verify_numbers.py")
    print("\nAll steps complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
