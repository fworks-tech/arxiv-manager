"""Draft configuration: model settings, timeouts, retry behavior."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Per-model token and timeout tuning."""
    max_tokens_easy: int = 4000
    max_tokens_hard: int = 16000
    timeout_easy: int = 120
    timeout_hard: int = 240


@dataclass
class DraftConfig:
    default_model: str = "minimax-m3"
    api_url: str = "https://opencode.ai/zen/go/v1/chat/completions"
    retries: int = 3
    thumbnail_size: tuple[int, int] = (1024, 1024)
    jpeg_quality: int = 85

    models: dict[str, ModelConfig] = field(default_factory=lambda: {
        "kimi": ModelConfig(
            max_tokens_easy=4000, max_tokens_hard=32000,
            timeout_easy=180, timeout_hard=300,
        ),
        "minimax": ModelConfig(
            max_tokens_easy=4000, max_tokens_hard=16000,
            timeout_easy=120, timeout_hard=240,
        ),
    })

    fallback: ModelConfig = field(default_factory=lambda: ModelConfig(
        max_tokens_easy=4000, max_tokens_hard=8000,
        timeout_easy=120, timeout_hard=180,
    ))

    def get_model_config(self, model_id: str) -> ModelConfig:
        """Look up config by model ID substring match."""
        for key, cfg in self.models.items():
            if key in model_id.lower():
                return cfg
        return self.fallback


CONFIG = DraftConfig()
