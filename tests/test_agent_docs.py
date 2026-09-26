"""The agent instructions (Claude skill, AGENTS.md, Cursor rule, Gemini extension) stay in sync."""

import json
import re
from pathlib import Path

import pytest

from trainjudge import __version__

ROOT = Path(__file__).parent.parent
AGENT_DOCS = [
    ROOT / "skills" / "trainjudge" / "SKILL.md",
    ROOT / "AGENTS.md",
    ROOT / ".cursor" / "rules" / "trainjudge.mdc",
]
# Every agent must learn these: the workflow, the data-safety rule, custom scorers,
# the verdicts, and how to keep the user posted.
REQUIRED = [
    "trainjudge diagnose",
    "trainjudge train",
    "trainjudge verify",
    "--mask-sensitive",
    "--allow-sensitive-data",
    "--replay",
    "--scorer",
    "IMPROVED",
    "REGRESSED",
    "REJECTED",
    "--watch --milestones",
    'pip install "trainjudge[mlx]"',
]


@pytest.mark.parametrize("path", AGENT_DOCS, ids=lambda p: p.name)
def test_agent_docs_cover_the_workflow(path):
    text = path.read_text(encoding="utf-8")
    assert [r for r in REQUIRED if r not in text] == []
    assert "git+https" not in text  # installs come from PyPI


def test_cursor_rule_is_agent_requested():
    text = (ROOT / ".cursor" / "rules" / "trainjudge.mdc").read_text(encoding="utf-8")
    front = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert front, "Cursor rules need frontmatter"
    assert "description: Use when" in front.group(1)
    assert "alwaysApply: false" in front.group(1)


def test_gemini_extension_uses_the_skill_not_a_context_file():
    # Gemini CLI discovers skills/<name>/SKILL.md itself and loads it on demand; a
    # contextFileName pointing at it too would put it in every session's context.
    manifest = json.loads((ROOT / "gemini-extension.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "trainjudge"
    assert "contextFileName" not in manifest
    assert (ROOT / "skills" / "trainjudge" / "SKILL.md").is_file()


def test_versions_match():
    # tomllib needs Python 3.11; 3.10 is supported.
    pyproject = re.search(r'^version = "(.+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    gemini = json.loads((ROOT / "gemini-extension.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert pyproject.group(1) == plugin["version"] == gemini["version"] == __version__
    assert f"version-{__version__}-green" in readme
    assert f"## {__version__} " in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
