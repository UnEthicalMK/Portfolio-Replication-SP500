import sys
from pathlib import Path

# Make the repository root importable regardless of pytest's invocation cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
