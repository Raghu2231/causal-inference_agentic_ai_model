from __future__ import annotations

import requests

BASE_URL = "http://localhost:8000"


def upload(file_bytes: bytes, filename: str) -> dict:
    r = requests.post(f"{BASE_URL}/upload", files={"file": (filename, file_bytes)})
    r.raise_for_status()
    return r.json()


def get_checklist(file_id: str) -> dict:
    r = requests.get(f"{BASE_URL}/checklist/{file_id}")
    r.raise_for_status()
    return r.json()


def get_eda(file_id: str, metric_group: str = "Suggestions", variable: str | None = None, include_zscore: bool = False) -> dict:
    params = {"metric_group": metric_group, "include_zscore": include_zscore}
    if variable:
        params["variable"] = variable
    r = requests.get(f"{BASE_URL}/eda/{file_id}", params=params)
    r.raise_for_status()
    return r.json()


def run_model(file_id: str, scenario_multiplier: float = 1.0, isolate_channel: str | None = None) -> dict:
    r = requests.post(
        f"{BASE_URL}/run/{file_id}",
        json={"scenario_multiplier": scenario_multiplier, "isolate_channel": isolate_channel},
    )
    r.raise_for_status()
    return r.json()
