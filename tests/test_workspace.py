from literature_rag.settings import DOWNLOAD_TIERS
from literature_rag.workspace import archive_previous_output, get_workspace


def test_workspace_is_normalized_and_deterministic(tmp_path):
    first = get_workspace(" Graph Neural Networks ", tmp_path)
    second = get_workspace("graph   neural networks", tmp_path)
    other = get_workspace("different topic", tmp_path)

    assert first == second
    assert first.root != other.root
    assert first.papers.is_dir()


def test_workspace_creates_one_folder_per_download_tier(tmp_path):
    workspace = get_workspace("federated intrusion detection", tmp_path)

    for tier in DOWNLOAD_TIERS:
        assert workspace.tier_dir(tier).is_dir()
    assert workspace.tier_dir("core").name == "core"
    assert not (workspace.papers / "excluded").exists()
    assert workspace.classification_cache.name == "classification.json"
    assert workspace.plan_cache.name == "search_plan.json"


def test_archive_previous_output_moves_only_report_artifacts(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "report.md").write_text("old report", encoding="utf-8")
    (output / "review.json").write_text("{}", encoding="utf-8")
    (output / "run.log").write_text("keep me", encoding="utf-8")

    destination = archive_previous_output(output)

    assert destination is not None
    assert (destination / "report.md").read_text(encoding="utf-8") == "old report"
    assert (destination / "review.json").exists()
    assert not (output / "report.md").exists()
    assert (output / "run.log").read_text(encoding="utf-8") == "keep me"
    assert archive_previous_output(output) is None
