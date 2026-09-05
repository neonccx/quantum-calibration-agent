"""Local, credential-free configuration. No provider login or automatic downloads."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import math
import os


@dataclass(frozen=True)
class Settings:
    policy: str = "rule"
    model: str | None = None
    adapter: str | None = None
    trust_remote_code: bool = False
    max_input_tokens: int = 20480
    max_new_tokens: int = 1024
    chat_max_new_tokens: int = 2048
    decode_backend: str = "auto"
    prompt_profile: str = "skill"
    fit_update_tool: bool = False
    request_timeout: float = 180.0
    seed: int = 20260903
    noise_scale: float = 1.0
    backend: str = "physical"
    simulation_profile: str = "nominal"
    max_steps: int = 30
    max_tool_calls: int = 8

    @classmethod
    def from_dict(cls, value: dict):
        if not isinstance(value, dict) or set(value) - {item.name for item in fields(cls)}:
            raise ValueError("Unknown config keys; credentials/API keys are not supported")
        result = cls(**value)
        if result.backend not in ("physical", "legacy") or result.simulation_profile not in ("nominal", "drift", "quasistatic", "ambiguous"):
            raise ValueError("Invalid simulation backend/profile")
        if result.backend == "legacy" and result.simulation_profile != "nominal":
            raise ValueError("Legacy simulator supports only nominal profile")
        if type(result.fit_update_tool) is not bool:
            raise ValueError("fit_update_tool must be boolean")
        if result.policy not in ("rule", "hf") or type(result.trust_remote_code) is not bool:
            raise ValueError("Invalid policy/trust_remote_code")
        if result.decode_backend not in ("auto", "hf", "cuda_graph"):
            raise ValueError("Invalid decode_backend")
        if result.prompt_profile not in ("minimal", "skill"):
            raise ValueError("Invalid prompt_profile")
        for key, low, high in (("max_input_tokens", 1, 131072), ("max_new_tokens", 1, 8192),
                               ("chat_max_new_tokens", 1, 4096),
                               ("max_steps", 1, 1000), ("max_tool_calls", 1, 100), ("seed", 0, 2**63 - 1)):
            value = getattr(result, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} must be an integer in [{low}, {high}]")
        for key, low, high in (("noise_scale", 0, 100), ("request_timeout", 1, 3600)):
            value = getattr(result, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid {key}")
        for key in ("model", "adapter"):
            value = getattr(result, key)
            if value is not None and (not isinstance(value, str) or not Path(value).is_absolute()):
                raise ValueError(f"{key} must be an absolute local directory path")
        if result.policy == "hf" and not result.model:
            raise ValueError("HF policy requires a local model directory")
        if result.policy == "rule" and (result.model or result.adapter or result.trust_remote_code or result.prompt_profile != "skill"):
            raise ValueError("Rule mode cannot silently ignore model options")
        return result

    def to_dict(self):
        return asdict(self)


def app_home(value=None) -> Path:
    return Path(value or os.environ.get("QM_AGENT_HOME") or Path.home() / ".local/share/qm-agent").expanduser().resolve()


def load_settings(home: Path) -> Settings:
    from .storage import read_json
    path = home / "config.json"
    return Settings.from_dict(read_json(path)) if path.exists() else Settings()


def save_settings(home: Path, settings: Settings):
    from .storage import atomic_json
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_json(home / "config.json", settings.to_dict(), replace=True)
