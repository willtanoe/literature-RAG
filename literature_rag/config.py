from __future__ import annotations

import getpass
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import keyring
from keyring.errors import KeyringError

from literature_rag.__log__ import get_logger
from literature_rag.resilience import atomic_write_json
from literature_rag.settings import CONFIG_PATH

logger = get_logger(__name__)

KEYRING_SERVICE = "agentic-literature-rag"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model_name: str


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not config_path.exists():
        return {"profiles": []}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {config_path}: {exc}") from exc
    if not isinstance(config.get("profiles"), list):
        raise RuntimeError(f"Invalid config format in {config_path}.")
    return config


def save_config(config: dict[str, Any], config_path: Path = CONFIG_PATH) -> None:
    try:
        atomic_write_json(config_path, config, 0o600)
    except OSError as exc:
        raise RuntimeError(f"Cannot save {config_path}: {exc}") from exc


def prompt_required(label: str, secret: bool = False) -> str:
    while True:
        value = (getpass.getpass(f"{label}: ") if secret else input(f"{label}: ")).strip()
        if value:
            return value
        logger.warning(f"{label} cannot be empty")


def add_profile(config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], str]:
    logger.info("Add OpenAI-compatible endpoint")
    name = prompt_required("Profile name")
    base_url = prompt_required("Base URL (for example https://host.example/v1)").rstrip("/")
    parsed_url = urlparse(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Base URL must be a valid HTTP(S) URL.")
    api_key = prompt_required("API key", secret=True)
    key_ref = str(uuid.uuid4())
    try:
        keyring.set_password(KEYRING_SERVICE, key_ref, api_key)
    except KeyringError as exc:
        key_ref = ""
        logger.debug(f"Keyring unavailable; session-only key: {exc}")
    model_name = prompt_required("Model name")
    profile = {
        "name": name,
        "base_url": base_url,
        "key_ref": key_ref,
        "models": [model_name],
    }
    config["profiles"].append(profile)
    save_config(config, config_path)
    profile["_session_api_key"] = api_key
    logger.info(f"Saved {name} in {config_path}")
    return profile, model_name


def add_model(config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], str]:
    profiles = config["profiles"]
    logger.info("Choose endpoint")
    for index, profile in enumerate(profiles, start=1):
        print(f"  {index}. {profile['name']} ({profile['base_url']})")
    choice = input("Endpoint number: ").strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(profiles):
        raise ValueError("Invalid endpoint selection.")
    profile = profiles[int(choice) - 1]
    model_name = prompt_required("Model name")
    models = profile.setdefault("models", [])
    if model_name not in models:
        models.append(model_name)
        save_config(config, config_path)
    logger.info(f"Model ready: {profile['name']} / {model_name}")
    return profile, model_name


def choose_llm(config_path: Path = CONFIG_PATH) -> LLMConfig:
    config = load_config(config_path)
    if not config["profiles"]:
        profile, model_name = add_profile(config, config_path)
        return _to_llm_config(profile, model_name, config, config_path)

    choices = [
        (profile, model) for profile in config["profiles"] for model in profile.get("models", [])
    ]
    logger.info("Choose LLM")
    for index, (profile, model) in enumerate(choices, start=1):
        print(f"  {index}. {profile['name']} / {model}")
    print(f"  {len(choices) + 1}. Add endpoint + API key")
    print(f"  {len(choices) + 2}. Add model to endpoint")
    selected = input("Selection [1]: ").strip() or "1"
    if not selected.isdigit():
        raise ValueError("Invalid LLM selection.")
    selected_index = int(selected)
    if 1 <= selected_index <= len(choices):
        profile, model_name = choices[selected_index - 1]
    elif selected_index == len(choices) + 1:
        profile, model_name = add_profile(config, config_path)
    elif selected_index == len(choices) + 2:
        profile, model_name = add_model(config, config_path)
    else:
        raise ValueError("Invalid LLM selection.")
    logger.info(f"Using {profile['name']} / {model_name}")
    return _to_llm_config(profile, model_name, config, config_path)


def _to_llm_config(
    profile: dict[str, Any],
    model_name: str,
    config: dict[str, Any],
    config_path: Path,
) -> LLMConfig:
    key_ref = profile.get("key_ref")
    session_key = profile.pop("_session_api_key", None)
    legacy_key = profile.pop("api_key", None)
    if legacy_key:
        key_ref = key_ref or str(uuid.uuid4())
        try:
            keyring.set_password(KEYRING_SERVICE, key_ref, legacy_key)
        except KeyringError as exc:
            key_ref = ""
            session_key = legacy_key
            logger.debug(f"Keyring unavailable; migrated key is session-only: {exc}")
        profile["key_ref"] = key_ref
        save_config(config, config_path)
        if key_ref:
            logger.info("Migrated API key to OS keyring")
    try:
        api_key = session_key or (
            keyring.get_password(KEYRING_SERVICE, key_ref) if key_ref else None
        )
    except KeyringError as exc:
        logger.warning(f"Keyring unavailable; requesting session key: {exc}")
        api_key = None
    if not api_key:
        api_key = prompt_required("API key (keyring unavailable or entry missing)", secret=True)
    values = LLMConfig(
        base_url=profile["base_url"],
        api_key=api_key,
        model_name=model_name,
    )
    if not all(asdict(values).values()):
        raise RuntimeError("Selected LLM profile is incomplete.")
    return values
