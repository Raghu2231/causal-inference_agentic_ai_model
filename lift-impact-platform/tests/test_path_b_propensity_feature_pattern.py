from __future__ import annotations

import ast
from pathlib import Path
import re


def test_path_b_propensity_feature_pattern_handles_column_casing() -> None:
    """Path B propensity regex patterns should match both uppercase and lowercase confound columns."""
    engineer_file = Path("Causal_Inference/utils/ml/engineer.py")
    source = engineer_file.read_text()
    tree = ast.parse(source)

    patterns = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PathBPropensityEngineer":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and getattr(item.target, "id", None) == "FEATURE_COLUMN_PATTERNS":
                    patterns = [ast.literal_eval(element) for element in item.value.elts]
                    break
    assert patterns, "Could not locate PathBPropensityEngineer feature patterns."

    assert any(re.match(pattern, "confound_path_b_channel_a") for pattern in patterns)
    assert any(re.match(pattern, "CONFOUND_PATH_B_CHANNEL_A") for pattern in patterns)
