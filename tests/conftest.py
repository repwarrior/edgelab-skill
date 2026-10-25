"""Make the skill's package importable and keep tests away from ~/.edgelab."""
import os
import sys
import tempfile
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent / "skills" / "edgelab"
sys.path.insert(0, str(SKILL_DIR))

# Must be set before edgelab.config is imported anywhere.
os.environ["EDGELAB_HOME"] = tempfile.mkdtemp(prefix="edgelab-test-")
