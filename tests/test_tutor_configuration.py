"""Static checks: repo tutor configuration stays honest and student-first."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_custom_agent_file_and_frontmatter():
    text = (ROOT / ".github/agents/apt-tutor.agent.md").read_text()
    assert text.startswith("---\n")
    assert "name: APT Tutor" in text
    assert "TODO(student)" in text
    assert "progress.yaml" in text


def test_progress_starts_unassessed():
    text = (ROOT / "tutor/progress.yaml").read_text()
    assert "status: not_assessed" in text
    assert "completed_labs: []" in text
    assert "last_confirmed_at: null" in text


def test_student_ownership_is_repo_wide():
    for path in ["AGENTS.md", ".github/copilot-instructions.md"]:
        text = (ROOT / path).read_text()
        assert "TODO" in text
    assert "learner" in (ROOT / "tutor/README.md").read_text().lower()
