import os
from pathlib import Path, PosixPath
from dataclasses import dataclass
from typing import TypeVar, Type
from pyspark.sql.column import Column
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
import pandas as pd

from pyspark.sql import DataFrame as SparkDataFrame
from mlops_suggestion_measurement.suggestions_measurement.config.ml import ModelConfig as ModelConfigObject
# FIXME: Creates circular import
# from suggestions_measurement.utils.ml import (
    # FeatureEngineer as FeatureEngineerObject,
    # FeatureEngineerValidator as FeatureEngineerValidatorObject,
    # ModelTrainer as ModelTrainerObject
# )

try:
    DataFrame: Type = TypeVar(
        'DataFrameLike',  
        pd.DataFrame,  # This has got to go - we don't respect the pd.DataFrame type.
        SparkDataFrame
    )
except NameError:  # without spark this should just raise errors - we're not prepared to handle it. 
    DataFrame: Type = TypeVar(
        'DataFrameLike',
        pd.DataFrame,
        'SparkDataFrame'
    )
# FeatureEngineer: Type = TypeVar(
#     'FeatureEngineerType',
#     'FeatureEngineer'
# )
Column: Type = TypeVar(
    'ColumnType',
    bound=Column  # Why not just typehint with pyspark.sql.Column?
)
ModelConfig: Type = TypeVar(
    'ModelConfigType', 
    bound=ModelConfigObject  # Same as above. 
)
PathLike: Type = TypeVar(
    # TODO not all the functions/methods that use this actually support all the types here.
    # FIXME we use the config.paths.Path class often - which is not included in this TypeVar
    'PathLike', 
    str, 
    bytes, 
    os.PathLike,
    Path,
    PosixPath
)
Self: Type = TypeVar('Self')  # This raises errors with pylance. 

@dataclass
class Types:
    Column: Type = Column
    DataFrame: Type = DataFrame
    # FeatureEngineer: Type = FeatureEngineer
    ModelConfig: Type = ModelConfig
    PandasDataFrame: Type = pd.DataFrame
    PathLike: Type = PathLike
    Self: Type = Self
    SparkDataFrame: Type = SparkDataFrame
    SparkSession: Type = SparkSession
    StringType: Type = StringType