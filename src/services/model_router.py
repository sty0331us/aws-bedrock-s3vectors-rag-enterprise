"""Intelligent foundation-model routing across the Claude 5 family.

Complex / multimodal prompts go to Claude 5 Sonnet (`us.anthropic.claude-sonnet-5`).
Short factual text prompts go to the fast/cost-efficient Haiku companion
(`us.anthropic.claude-haiku-5`). When `BEDROCK_PROMPT_ROUTER_ARN` is set,
Bedrock's managed Anthropic prompt router makes the decision instead of
these local heuristics.
"""

from __future__ import annotations

import re

from api.schemas import RagQueryRequest
from config import Settings

_COMPLEXITY_HINTS = re.compile(
    r"\b(compare|contrast|analyze|explain in detail|step by step|why|trade-?off|"
    r"architect|design|reason|evaluate|summarize the differences)\b",
    re.IGNORECASE,
)


class ModelRouter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def resolve(self, request: RagQueryRequest) -> str:
        if self._settings.bedrock_prompt_router_arn:
            return self._settings.bedrock_prompt_router_arn
        if request.image is not None:
            return self._settings.claude_sonnet_model_id
        text = request.text or ""
        if len(text) > 1_200 or _COMPLEXITY_HINTS.search(text):
            return self._settings.claude_sonnet_model_id
        return self._settings.claude_haiku_model_id
