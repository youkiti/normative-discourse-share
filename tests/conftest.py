"""Make the standalone speaker scripts importable in synthetic tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/quote_check"))
