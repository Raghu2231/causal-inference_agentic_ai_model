from __future__ import annotations

from backend.services.insight_generator import InsightGenerator


def test_insight_generator_fallback_has_core_values() -> None:
    summary = {
        "path_a": {
            "aggregated_incremental_actions": 25.0,
            "channel_lift": {"action_email": 10.0, "action_call": 15.0},
        },
        "path_b": {
            "aggregated_incremental_trx": 40.0,
            "aggregated_incremental_nbrx": 12.0,
        },
    }
    generator = InsightGenerator()
    result = generator.generate(summary)

    assert "incremental actions" in result.narrative.lower()
    assert len(result.bullets) >= 3
    assert result.source in {"rule_based", "llm"}
