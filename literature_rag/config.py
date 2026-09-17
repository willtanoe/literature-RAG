from __future__ import annotations

import getpass
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import keyring
from keyring.errors import KeyringError

from literature_rag.resilience import atomic_write_json, redact_secrets
from literature_rag.settings import CONFIG_PATH

KEYRING_SERVICE = "agentic-literature-rag"
S2_KEYRING_USER = "semantic-scholar"
S2_ENV_VAR = "S2_API_KEY"


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
        print(f"Configuration -> {label} cannot be empty")


def add_profile(config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], str]:
    print("\nAdd OpenAI-compatible endpoint")
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
        print(
            f"Configuration -> keyring unavailable; key will be kept for this session only: {exc}"
        )
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
    print(f"Configuration -> saved {name} in {config_path}")
    return profile, model_name


def add_model(config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], str]:
    profiles = config["profiles"]
    print("\nChoose endpoint")
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
    print(f"Configuration -> model ready: {profile['name']} / {model_name}")
    return profile, model_name


def _confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N]: ").strip().lower() in {"y", "yes"}


def remove_model(config: dict[str, Any], config_path: Path) -> bool:
    choices = [
        (profile, model) for profile in config["profiles"] for model in profile.get("models", [])
    ]
    if not choices:
        print("Configuration -> no models are available to remove")
        return False
    print("\nRemove model")
    for index, (profile, model) in enumerate(choices, start=1):
        print(f"  {index}. {profile['name']} / {model}")
    selected = input("Model number: ").strip()
    if not selected.isdigit() or not 1 <= int(selected) <= len(choices):
        raise ValueError("Invalid model selection.")
    profile, model_name = choices[int(selected) - 1]
    if not _confirm(f"Remove model {profile['name']} / {model_name}?"):
        print("Configuration -> model removal cancelled")
        return False
    profile["models"].remove(model_name)
    save_config(config, config_path)
    print(f"Configuration -> removed model: {profile['name']} / {model_name}")
    if not profile["models"]:
        print(f"Configuration -> endpoint retained without models: {profile['name']}")
    return True


def remove_endpoint(config: dict[str, Any], config_path: Path) -> bool:
    profiles = config["profiles"]
    if not profiles:
        print("Configuration -> no endpoints are available to remove")
        return False
    print("\nRemove endpoint")
    for index, profile in enumerate(profiles, start=1):
        models = ", ".join(profile.get("models", [])) or "no models"
        print(f"  {index}. {profile['name']} ({profile['base_url']}) [{models}]")
    selected = input("Endpoint number: ").strip()
    if not selected.isdigit() or not 1 <= int(selected) <= len(profiles):
        raise ValueError("Invalid endpoint selection.")
    profile = profiles[int(selected) - 1]
    if not _confirm(
        f"Remove endpoint {profile['name']} and all {len(profile.get('models', []))} model(s)?"
    ):
        print("Configuration -> endpoint removal cancelled")
        return False
    key_ref = profile.get("key_ref")
    profiles.remove(profile)
    save_config(config, config_path)
    if key_ref:
        try:
            keyring.delete_password(KEYRING_SERVICE, key_ref)
        except KeyringError as exc:
            print(f"Configuration -> endpoint removed; keyring cleanup unavailable: {exc}")
    print(f"Configuration -> removed endpoint: {profile['name']}")
    return True


def get_semantic_scholar_key() -> str:
    import os

    env_key = os.environ.get(S2_ENV_VAR, "").strip()
    if env_key:
        print("Configuration -> using Semantic Scholar API key from environment")
        return env_key
    try:
        stored = keyring.get_password(KEYRING_SERVICE, S2_KEYRING_USER) or ""
    except KeyringError:
        stored = ""
    if stored:
        print("Configuration -> using stored Semantic Scholar API key")
        return stored
    try:
        value = input(
            "Semantic Scholar API key [optional, Enter to skip, "
            f"or set the {S2_ENV_VAR} environment variable]: "
        ).strip()
    except EOFError:
        return ""
    if not value:
        return ""
    try:
        keyring.set_password(KEYRING_SERVICE, S2_KEYRING_USER, value)
        print("Configuration -> stored Semantic Scholar API key in the keyring")
    except KeyringError as exc:
        print(f"Configuration -> keyring unavailable; key is session-only: {exc}")
    return value


def check_endpoint(llm_config: LLMConfig) -> None:
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    print("Configuration -> checking endpoint availability")
    started = time.monotonic()
    llm = ChatOpenAI(
        base_url=llm_config.base_url,
        api_key=SecretStr(llm_config.api_key),
        model=llm_config.model_name,
        temperature=0,
        timeout=30,
    )
    try:
        response = llm.invoke("Reply with the single word: OK")
    except Exception as exc:
        raise ValueError(
            "Endpoint check failed for "
            f"{llm_config.model_name} at {llm_config.base_url}: "
            f"{redact_secrets(exc, (llm_config.api_key,))}"
        ) from exc
    if not str(response.content).strip():
        raise ValueError("Endpoint check failed: the model returned an empty response.")
    print(f"Configuration -> endpoint check passed in {time.monotonic() - started:.1f}s")


def choose_llm(config_path: Path = CONFIG_PATH) -> LLMConfig:
    config = load_config(config_path)
    if not config["profiles"]:
        profile, model_name = add_profile(config, config_path)
        return _to_llm_config(profile, model_name, config, config_path)
    while True:
        choices = [
            (profile, model)
            for profile in config["profiles"]
            for model in profile.get("models", [])
        ]
        print("\nChoose LLM")
        for index, (profile, model) in enumerate(choices, start=1):
            print(f"  {index}. {profile['name']} / {model}")
        add_endpoint_index = len(choices) + 1
        add_model_index = len(choices) + 2
        remove_model_index = len(choices) + 3
        remove_endpoint_index = len(choices) + 4
        print(f"  {add_endpoint_index}. Add endpoint + API key")
        print(f"  {add_model_index}. Add model to endpoint")
        print(f"  {remove_model_index}. Remove model")
        print(f"  {remove_endpoint_index}. Remove endpoint")
        prompt = "Selection [1]: " if choices else "Selection: "
        selected = input(prompt).strip() or ("1" if choices else "")
        if not selected.isdigit():
            raise ValueError("Invalid LLM selection.")
        selected_index = int(selected)
        if 1 <= selected_index <= len(choices):
            profile, model_name = choices[selected_index - 1]
            print(f"Configuration -> using {profile['name']} / {model_name}")
            return _to_llm_config(profile, model_name, config, config_path)
        if selected_index == add_endpoint_index:
            profile, model_name = add_profile(config, config_path)
            print(f"Configuration -> using {profile['name']} / {model_name}")
            return _to_llm_config(profile, model_name, config, config_path)
        if selected_index == add_model_index:
            profile, model_name = add_model(config, config_path)
            print(f"Configuration -> using {profile['name']} / {model_name}")
            return _to_llm_config(profile, model_name, config, config_path)
        if selected_index == remove_model_index:
            remove_model(config, config_path)
            continue
        if selected_index == remove_endpoint_index:
            remove_endpoint(config, config_path)
            if not config["profiles"]:
                print("Configuration -> no endpoints remain; add one to continue")
                profile, model_name = add_profile(config, config_path)
                return _to_llm_config(profile, model_name, config, config_path)
            continue
        raise ValueError("Invalid LLM selection.")


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
            print(f"Configuration -> keyring unavailable; migrated key is session-only: {exc}")
        profile["key_ref"] = key_ref
        save_config(config, config_path)
        if key_ref:
            print("Configuration -> migrated API key to the operating-system keyring")
    try:
        api_key = session_key or (
            keyring.get_password(KEYRING_SERVICE, key_ref) if key_ref else None
        )
    except KeyringError as exc:
        print(f"Configuration -> keyring unavailable; requesting a session key: {exc}")
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
