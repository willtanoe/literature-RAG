"""Throwaway driver for the top_n=8 smoke run. Delete after the run."""

from pathlib import Path

from literature_rag.config import _to_llm_config, check_endpoint, load_config
from literature_rag.settings import CONFIG_PATH

PROFILE = "gk"
MODEL = "gpt-5.6-sol"
TOPIC = "Differential Privacy Federated Learning Intrusion Detection System"
OBJECTIVE = "build a matrix of methods, datasets, gaps, and future work."
TOP_N = 8
SMOKE_DIR = Path("./.smoke_run")


def _llm_config():
    config = load_config()
    profile = next(item for item in config["profiles"] if item["name"] == PROFILE)
    return _to_llm_config(profile, MODEL, config, CONFIG_PATH)


def main() -> None:
    import os

    from literature_rag.pipeline import run_pipeline

    llm_config = _llm_config()
    check_endpoint(llm_config)
    report, _ = run_pipeline(
        topic=TOPIC,
        analysis_question=OBJECTIVE,
        llm_config=llm_config,
        top_n=TOP_N,
        retrieval_k=max(12, min(TOP_N * 2, 64)),
        download_dir=SMOKE_DIR,
        year_min=2021,
        year_max=2026,
        s2_api_key=os.environ.get("S2_API_KEY", ""),
    )
    print(f"\nSmoke run finished | report characters: {len(report):,}")


if __name__ == "__main__":
    main()
