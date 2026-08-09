import sys
from pathlib import Path

# make the repo root importable regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent))
