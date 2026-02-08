import os
from typing import Dict, Sequence
from .types import Types
from .columns import Column
from .ml import ALLOWED_MODEL_OBJECTS, ALLOWED_MODEL_TYPES, FEATURE_CONTRIBUTION_EXTRACTORS, ModelConfig
from .paths import PathConfig, HDFSPath
# from .plots import Colors
from .hopper import HopperOutput
from .archivist import (
    S3_TARGET, 
    ARCHIVIST_HDFS_OUTPUT_DIR, 
    ARCHIVIST_OUTPUT_DIR
)
os.environ['SPARK_HOME']: str = '/usr/lib/spark'
os.environ['HADOOP_HOME']: str = '/usr/lib/hadoop'

DEFAULT_ADSTOCK_PARAMS: Dict[str, Dict[str, float]] = {
    "suggestion_count": {
        "type": "exponential",
        "decay_rate": 0.3,
        "adstock_steps": 4,
    },
    "percent_other_hcps_suggested": {
        "type": "exponential",
        "decay_rate": 0.3,
        "adstock_steps": 4,
    },
    # "propensity_score_path_A_suggestion_count": {
    #     "type": "exponential",
    #     "decay_rate": 0.3,
    #     "adstock_steps": 4
    # },
    "action_count": {
        "type": "exponential",
        "decay_rate": 0.69,
        "adstock_steps": 4,
    },
    # 'propensity_score_path_B_action_count': {
    #     "type": "exponential",
    #     "decay_rate": 0.06,
    #     "adstock_steps": 10,
    # },
}

__all__: Sequence[str] = [
    "ALLOWED_MODEL_OBJECTS",
    "ALLOWED_MODEL_TYPES",
    "Colors",
    "Column",
    "DEFAULT_ADSTOCK_PARAMS",
    "HDFSPath",
    "HopperOutput",
    "ModelConfig",
    "PathConfig",
    "Types",
]
