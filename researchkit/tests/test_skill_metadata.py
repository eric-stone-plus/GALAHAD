from pathlib import Path


# Wave A finance skills were stripped from the public GALAHAD tree
# (08-18 commercial boundary). This test pins that product decision.
SKILLS_ROOT = Path(__file__).parents[2] / ".agents" / "skills"


def test_public_tree_does_not_ship_wave_a_finance_skills() -> None:
    assert not SKILLS_ROOT.exists(), (
        "Wave A finance plugins must stay out of the public tree; "
        f"found {SKILLS_ROOT}"
    )
