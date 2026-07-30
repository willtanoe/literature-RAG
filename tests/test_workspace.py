from literature_rag.workspace import get_workspace


def test_workspace_is_normalized_and_deterministic(tmp_path):
    first = get_workspace(" Graph Neural Networks ", tmp_path)
    second = get_workspace("graph   neural networks", tmp_path)
    other = get_workspace("different topic", tmp_path)

    assert first == second
    assert first.root != other.root
    assert first.papers.is_dir()
