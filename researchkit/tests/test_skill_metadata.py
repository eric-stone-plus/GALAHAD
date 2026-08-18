import re
from pathlib import Path


SKILLS_ROOT = Path(__file__).parents[2] / ".agents" / "skills"


def test_wave_a_skill_metadata_and_references_are_complete() -> None:
    expected = {
        "plan-investment-research",
        "audit-data-provenance",
        "research-sector-structure",
        "analyze-financial-statements",
        "run-dcf-valuation",
        "run-comps-valuation",
    }
    assert {path.name for path in SKILLS_ROOT.iterdir() if path.is_dir()} == expected
    for name in expected:
        skill_dir = SKILLS_ROOT / name
        skill_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        yaml_text = (skill_dir / "agents/openai.yaml").read_text(encoding="utf-8")
        assert f"name: {name}" in skill_text
        assert "TODO" not in skill_text
        assert f"${name}" in yaml_text
        assert "dependencies:" not in yaml_text
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", skill_text):
            if not target.startswith(("http://", "https://", "#")):
                assert (skill_dir / target.split("#", 1)[0]).exists(), (name, target)
