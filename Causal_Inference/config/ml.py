from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Callable, List

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression, GBTRegressor

from .paths import PathConfig

# TODO: Make sure that this allows for creations of multiple instances of each
# with out pointing to the same memory space.
ALLOWED_MODEL_OBJECTS: Dict[str, object] = {
    'LogisticRegression': LogisticRegression,
    'LinearRegression': LinearRegression,
    'GBTRegressor': GBTRegressor,
}
ALLOWED_MODEL_TYPES: Sequence[str] = (
    'PathA_PropensityModel',
    'PathB_PropensityModel',
    'PathA_TreatmentEffectModel',
    'PathB_TreatmentEffectModel',
)
ALLOWED_FEATURE_PATTERNS: Sequence[str] = (
    # Matches 'confound_path_{path}_*'
    r'^confound_path_(A|B)_.*', 
    # Matches 'covariate_path{path}_*' 
    r'^covariate_path_(A|B)_.*$',
    # Matches 'propensity_score_path_{path}_{some_suggestion_type}'
    r'^propensity_score_path_(A|B)_[^_]+$'
)
FEATURE_CONTRIBUTION_EXTRACTORS: Dict[str, Callable] = {
    'LogisticRegressionModel': lambda model: model.coefficients,
    'LinearRegressionModel': lambda model: model.coefficients,
    'GBTRegressionModel': lambda model: model.featureImportances.toArray().tolist(),
}

REP_BASED_FEATURES: List[str] = [
    'covariate_path_A_rep_region', 
    'covariate_path_A_bus_sales_force',
    'covariate_path_A_rep_gender', 
    'covariate_path_A_rep_district', 
    'covariate_path_A_sample_mgnt_ind', 
    'covariate_path_A_territory_no', 
    'covariate_path_A_category_descr', 
    'covariate_path_A_bus_sales_force',
]


@dataclass
class ModelConfig:

    # Unique Model Triangle
    suggestion_type: str
    action_type: str
    outcome_type: str

    # Filesystem Configuration
    path: PathConfig

    # Modeling-Path Specific
    propensity_column: str
    model_type: str

    # Metadata
    brand_ind: str
    run_id: str
    alloy: str
    model_id: str

    # Adstock Parameters
    adstock_params: Optional[Dict[str, Dict[str, float]]] = None

    # Treatment Effect Models need a Model ID for the propensity model
    dependent_model_id: Optional[str] = None

    # Metadata 
    start_date: Optional[str] = None 
    end_date: Optional[str] = None


### Model Path Info

# TODO delete me
# ModelPathInfo = namedtuple(
#     # This describes Path A and Path B models - not to be confused with filesystem paths.
#     'ModelPathInfo',
#     [
#         'modeling_path',
#         'treatment_variable'
#         'outcome_variable',
#     ]
# )

# PATH_A_INFO = ModelPathInfo(
#     'PathA',
#     'suggestion_count',
#     'action_count'
# )

# PATH_B_INFO = ModelPathInfo(
#     'PathB',
#     'action_count',
#     'suggestion_count'
# )