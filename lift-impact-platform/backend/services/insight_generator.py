from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib import request


@dataclass
class InsightResult:
    narrative: str
    bullets: List[str]
    source: str


class InsightGenerator:
    """Generates business-facing insights from model summary.

    Uses an LLM endpoint when configured, otherwise falls back to deterministic
    rule-based insights so the workflow remains fully functional offline.
    """

    def __init__(self) -> None:
        self.api_url = os.getenv("LLM_API_URL", "").strip()
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    def generate(self, summary: Dict[str, Any], prompt_context: str = "") -> InsightResult:
        fallback = self._fallback(summary)
        if not self.api_url:
            return InsightResult(narrative=fallback["narrative"], bullets=fallback["bullets"], source="rule_based")

        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a causal inference strategist. Summarize lift results into concise business actions.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps({"summary": summary, "context": prompt_context})[:12000],
                    },
                ],
                "temperature": 0.2,
            }
            req = request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
                },
                method="POST",
            )
            with request.urlopen(req, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
            text = (
                body.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not text:
                return InsightResult(narrative=fallback["narrative"], bullets=fallback["bullets"], source="rule_based")
            return InsightResult(
                narrative=text,
                bullets=fallback["bullets"],
                source="llm",
            )
        except Exception:
            return InsightResult(narrative=fallback["narrative"], bullets=fallback["bullets"], source="rule_based")

    @staticmethod
    def _fallback(summary: Dict[str, Any]) -> Dict[str, Any]:
        path_a = summary.get("path_a", {})
        path_b = summary.get("path_b", {})
        inc_actions = float(path_a.get("aggregated_incremental_actions", 0.0))
        inc_trx = float(path_b.get("aggregated_incremental_trx", 0.0))
        inc_nbrx = float(path_b.get("aggregated_incremental_nbrx", 0.0))
        channel_lift = path_a.get("channel_lift", {}) or {}
        best_channel = max(channel_lift, key=channel_lift.get) if channel_lift else "n/a"

        narrative = (
            f"Model results indicate +{inc_actions:.2f} incremental actions translating to "
            f"+{inc_trx:.2f} TRX and +{inc_nbrx:.2f} NBRX. "
            f"Top-performing channel appears to be '{best_channel}'."
        )
        bullets = [
            f"Prioritize channel mix toward {best_channel} where lift concentration is highest.",
            "Use scenario multiplier simulations before scaling field execution plans.",
            "Audit low-lift territories/HCP cohorts and rebalance suggestion intensity.",
        ]
        return {"narrative": narrative, "bullets": bullets}
