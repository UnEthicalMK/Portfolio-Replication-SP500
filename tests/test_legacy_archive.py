"""S0 acceptance: legacy artifact provenance (roadmap S0; finding D-11).

The two orphan CSVs were relocated from `data/` to `artifacts/legacy/` at the
S0 checkpoint (tracked Git renames; contents preserved). This test pins the
archived location and the provenance documentation.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "artifacts" / "legacy" / "PROVENANCE.md"
_ORPHANS = ("benchmark_returns_2023.csv", "portfolio_returns_2023.csv")


def test_provenance_document_exists():
    assert PROVENANCE.is_file()


def test_provenance_names_both_orphans_and_origin():
    doc = PROVENANCE.read_text(encoding="utf-8")
    for needle in (
        "benchmark_returns_2023.csv",
        "portfolio_returns_2023.csv",
        "03454bc",       # the commit that switched ^SP500TR -> ^GSPC
        "D-11",          # investigation-report finding
        "^SP500TR",      # documents the original total-return benchmark
        "artifacts/legacy",  # documents the archive location
    ):
        assert needle in doc, f"PROVENANCE.md must reference {needle!r}"


def test_orphans_archived_under_legacy_and_absent_from_data():
    """The orphans live in artifacts/legacy/ and nowhere else."""
    for name in _ORPHANS:
        assert (ROOT / "artifacts" / "legacy" / name).is_file(), (
            f"{name} must be archived under artifacts/legacy/"
        )
        assert not (ROOT / "data" / name).exists(), f"{name} must no longer sit in data/"
