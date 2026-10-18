"""Structural checks for the self-contained, learner-owned notebook curriculum."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


def test_all_notebooks_have_self_contained_material():
    assert len(NOTEBOOKS) == 16
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text())
        assert nb["nbformat"] == 4
        cells = nb["cells"]
        assert all(isinstance(c.get("source"), list) for c in cells)
        text = "\n".join("".join(c["source"]) for c in cells)
        assert "Worked example" in text or "worked numerical example" in text
        assert "TODO" in text
        assert "plot" in text.lower() or "graph" in text.lower()
        assert "gradient" in text.lower() or "verifier" in text.lower() or "evidence" in text.lower()
        assert not any(c.get("execution_count") is not None for c in cells if c["cell_type"] == "code")
