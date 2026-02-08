from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class RunRequest(BaseModel):
    scenario_multiplier: float = 1.0
    isolate_channel: Optional[str] = None


class RunResponse(BaseModel):
    summary: Dict[str, Any]
