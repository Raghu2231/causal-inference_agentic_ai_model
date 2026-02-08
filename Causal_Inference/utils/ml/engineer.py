'''
These classes contain the feature engineers for each of the ML models.

The 4 classes are:
- PathAPropensityEngineer
- PathBPropensityEngineer
- PathATreatmentEffectEngineer
- PathBTreatmentEffectEngineer

- TreatmentEffectEngineers have to capture predictions from Propensity models
- PathB models are indexed differently - and it is on the FeatureEngineer to handle this.

TODO:
  - Add missing docstring and test
'''

from abc import ABC, abstractmethod
from functools import reduce
from pathlib import PurePath
from typing import Dict, Optional, Sequence, TYPE_CHECKING
import re
from ..adstock import PathAAdstock, PathBAdstock

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
    StandardScaler
)
from pyspark.sql import SparkSession, functions as F

from ... import logger, config
from ..spark import PathAAggregator, PathBAggregator
from .smelter import (
    PercentOtherHCPSSuggestedSmelter,
    GlobalActivityIndexSmelter,
    RepActivityIndexSmelter,
    HCPActivityIndexSmelter
)


# FEATURE ENGINEERS

class FeatureEngineer(ABC):
    """
    Abstract base class for feature engineering in machine learning models.

    Attributes:
        df (config.Types.DataFrame): The input DataFrame containing the data.
        df_ (config.Types.DataFrame): The output DataFrame including the `features` column - suitable for ML training
        feature_columns (Optional[Sequence[str]]): The list of feature columns.
        assembler (Optional[VectorAssembler]): The VectorAssembler for assembling feature vectors.

    Methods:
        MODEL_TYPE (str): Abstract property that defines the model type.
        FEATURE_COLUMN_PATTERNS (Sequence[str]): Abstract property that defines the patterns for feature columns.
        _get_feature_columns(): Identifies and sets the feature columns based on the defined patterns.
        _get_vector_assembler(): Creates and sets the VectorAssembler for the feature columns.
        run() -> config.Types.DataFrame: Transforms the DataFrame using the VectorAssembler and returns the transformed DataFrame.
    """

    def __init__(
            self,
            df: config.Types.DataFrame,
            model_config: config.Types.ModelConfig,
            prop_pred_df=None
    ) -> None:

        logger.debug(
            'Initializing FeatureEngineer for model_type: %s',
            self.MODEL_TYPE,
        )

        self.df: config.Types.DataFrame = df
        self.model_config: config.ModelConfig = model_config
        self.adstocker: object = self.ADSTOCKER(model_config)

        (
            FeatureEngineerValidator(
                model_type=self.MODEL_TYPE,
                feature_column_patterns=self.FEATURE_COLUMN_PATTERNS
            )
            .run()
        )

        self.prop_pred_df = prop_pred_df
        self.df_: Optional[config.Types.DataFrame] = None
        self.feature_columns: Optional[Sequence[str]] = None
        self.assembler: Optional[Pipeline] = None

    @property
    @abstractmethod
    def MODEL_TYPE(self) -> str:
        pass

    @property
    @abstractmethod
    def FEATURE_COLUMN_PATTERNS(self) -> Sequence[str]:
        pass

    @property
    @abstractmethod
    def ADSTOCKER(self) -> object:
        pass

    @staticmethod
    def __get_categorical_encoder(
            inputCol: str,
            handleInvalid: str = 'skip'
    ) -> Pipeline:
        return Pipeline(
            stages=[
                StringIndexer(
                    inputCol=inputCol,
                    outputCol=f'{inputCol}_numeric',
                    handleInvalid=handleInvalid,
                ),
            ]
        )

    @staticmethod
    def __update_if_adstocking(
            feature_columns: Sequence[str],
    ) -> Sequence[str]:
        logger.debug('Checking for adstocked features.')
        adstock_features: Sequence[str] = [
            _column for _column in feature_columns
            if re.match(r'.*_adstock$', _column)
        ]
        if len(adstock_features) > 0:
            logger.debug('Found %d adstocked features: %s', len(adstock_features), adstock_features)
            feature_columns_: Sequence[str] = list(
                set(feature_columns) - set(map(
                    lambda _feature: _feature.split('_adstock')[0],
                    adstock_features
                ))
            )
            logger.debug('Final feature set: %s', feature_columns_)

            return feature_columns_

        logger.debug('No adstocked features found.')
        return feature_columns

    def _filter_model_triangle(self, df: config.Types.DataFrame) -> config.Types.DataFrame:

        logger.debug(
            'Filtering DataFrame for suggestion_type: %s, action_type: %s, outcome_type: %s',
            self.model_config.suggestion_type,
            self.model_config.action_type,
            self.model_config.outcome_type
        )
        _df: config.Types.DataFrame = (
            df
            .where(F.col("suggestion_type") == self.model_config.suggestion_type)
            .where(F.col("action_type") == self.model_config.action_type)
            .where(F.col("outcome_type") == self.model_config.outcome_type)
        )

        # assert _df.count() > 0, (
        #     f'No data for suggestion_type: {self.model_config.suggestion_type}, '
        #     f'action_type: {self.model_config.action_type}, '
        #     f'outcome_type: {self.model_config.outcome_type}'
        # )

        return _df

    @staticmethod
    def _filter_study_period(
            df: config.Types.DataFrame,
            start_date: Optional[str],
            end_date: Optional[str],
    ) -> config.Types.DataFrame:

        logger.debug(
            'Filtering DataFrame for study period: [%s, %s)',
            start_date,
            end_date,
        )

        _df = df
        if start_date:
            _df = _df.where(F.col("week_end_date") >= start_date)
        if end_date:
            _df = _df.where(F.col("week_end_date") < end_date)

        # logger.debug(
        #     "%d rows remaining in the dataset",
        #     _df.count()
        # )

        return _df
        

    def _get_feature_columns(self, df) -> None:
        """
        Extracts and sets the feature columns from the dataframe based on the specified pattern.

        Returns:
            None
        """
        # ToDO: Update with list from Ryan if needed
        # features_to_drop: Sequence[str] = [
        #     'confound_path_A_theme_skyrizi_stelara_increase_qualified',
        #     'confound_path_A_theme_skyrizi_social_proofing_qualified',
        #     'confound_path_A_theme_skyrizi_cosentyx_increase_qualified',
        #     'confound_path_A_theme_derm_ps_bime_increase_qualified',
        #     'confound_path_A_theme_derm_ps_nba_trigger_qualified',
        #     'confound_path_A_theme_ps_p2p_nba_trigger_qualified',
        #     'confound_path_A_theme_ps_psa_p2p_nba_trigger_qualified',
        #     'covariate_path_A_callplan_flag',
        #     'covariate_path_A_theme_ps_psa_p2p_nba_trigger',
        #     'covariate_path_A_theme_improvement_of_care',
        #     'covariate_path_A_theme_derm_ps_nba_trigger',
        #     'covariate_path_A_theme_ps_p2p_nba_trigger',
        #     'covariate_path_A_theme_skyrizi_social_proofing',
        #     'covariate_path_A_theme_skyrizi_cosentyx_increase',
        #     'covariate_path_A_theme_skyrizi_stelara_increase',
        #     'covariate_path_A_theme_derm_ps_bime_increase'
        # ]

        # logger.debug('Will exclude the following features if they exist: %s', features_to_drop)
        logger.debug('Getting feature columns using patterns: %s', self.FEATURE_COLUMN_PATTERNS)
        
        
        protected_columns: Sequence[str] = [
            'week_end_date',
            'rep_id',
            'abbott_customer_id',
            'suggestion_type',
            'suggestion_count',
            'suggestion_count_adstock',
            'action_type',
            'action_count',
            'outcome_type',
            'outcome_count',
            'percent_other_hcps_suggested',
            'percent_other_hcps_suggested_adstock',
        ]
        logger.debug('Using the following protected columns: %s', protected_columns)
        columns_to_check: Sequence[str] = (
            df
            .columns
        )
        logger.debug(
            'Looking for features in the following %d columns: %s',
            len(columns_to_check),
            columns_to_check
        )
        df = df.fillna(0, subset=columns_to_check)
        
        
        null_columns = (
            df
            .select([
                F.sum(F.isnull(F.col(c)).cast("int")).alias(c)
                for c in columns_to_check
            ])
            .toPandas()
            .melt(var_name="variable", value_name="null_count")
            .query("null_count == @df.count()")["variable"]
            .tolist()
        )
        logger.debug(f'Columns entirely null: {null_columns}')
        

        summary_stats = df.select(
            *[F.countDistinct(_column).alias(f"{_column}_distinct") for _column in columns_to_check],
        ).collect()[0]
        matched_columns: Sequence[str] = [
            _column for _column in set(columns_to_check + protected_columns)
            if any(re.match(pattern, _column) for pattern in self.FEATURE_COLUMN_PATTERNS)
        ]
        final_columns: Sequence[str] = [
            _column for _column in matched_columns
            # if not _column in features_to_drop
            # TODO: Drop columns in all null
            if summary_stats[f"{_column}_distinct"] > 1
               or _column in protected_columns
               and _column not in null_columns
        ]
        dropped_columns: Sequence[str] = list(
            set(matched_columns) - set(final_columns)
        )
        
        self.feature_columns: Sequence[str] = self.__update_if_adstocking(
            feature_columns=final_columns
        )

        logger.debug(
            'Found %d feature columns: %s',
            len(self.feature_columns),
            str(self.feature_columns)
        )
        logger.debug('Dropped %d columns: %s', len(dropped_columns), dropped_columns)
        
    def _convert_numeric_null_to_zero(
        self,
        df: config.Types.DataFrame
    ) -> config.Types.DataFrame:
        protected_columns: Sequence[str] = [
            'week_end_date',
            'rep_id',
            'abbott_customer_id',
            'suggestion_type',
            'suggestion_count',
            'suggestion_count_adstock',
            'action_type',
            'action_count',
            'outcome_type',
            'outcome_count',
            'percent_other_hcps_suggested',
            'percent_other_hcps_suggested_adstock',
        ]
 
        columns_to_check: Sequence[str] = (
            df
            .columns
        )
 
        matched_columns: Sequence[str] = [
                    _column for _column in set(columns_to_check + protected_columns)
                    if any(re.match(pattern, _column) for pattern in self.FEATURE_COLUMN_PATTERNS) 
                ]
 
        numeric_columns: Sequence[str] = [
            _column for _column in matched_columns
            if df.schema[_column].dataType != config.Types.StringType()
        ]
 
        df = df.fillna(0, subset=numeric_columns)
 
        # Step 2: Replace NaNs with 0
        logger.debug(f'Converting numeric columns nans to zero: {numeric_columns}')
        for c in numeric_columns:
            df = df.withColumn(c, when(isnan(col(c)), 0).otherwise(col(c)))
 
        return df

    def _make_vector_assembler(
            self,
            df,
            handleInvalid: str = 'skip'
    ):
        """
        Creates a VectorAssembler for the feature columns.

        Attributes:
            assembler (VectorAssembler): The VectorAssembler object that combines the feature columns into a single
            vector column.
        """

        if not self.feature_columns:
            self._get_feature_columns(df)

        string_columns: Sequence[str] = [
            _column for _column in self.feature_columns
            if df.schema[_column].dataType == config.Types.StringType()
        ]
        numeric_columns: Sequence[str] = [
            _column for _column in self.feature_columns
            if _column not in string_columns
        ]

        if (not string_columns) and (not numeric_columns):
            # TODO ? Maybe one day we should support mean-only prediction.
            raise RuntimeError('No feature columns found for VectorAssembler.')

        feature_schema: Dict[str, str] = {
            _column: df.schema[_column].dataType
            for _column in self.feature_columns
        }

        logger.debug(
            'Getting feature assembler for %d input columns: %s',
            len(self.feature_columns),
            self.feature_columns
        )
        logger.debug('Feature column schema: %s', feature_schema)

        stages = []
        if len(string_columns) > 0:
            logger.debug('Will one-hot encode the following %d columns: %s', len(string_columns), string_columns)
            logger.debug('Adding string indexers.')
            stages.extend([
                self.__get_categorical_encoder(
                    col,
                    handleInvalid=handleInvalid
                )
                for col in string_columns
            ])
            logger.debug('Adding one-hot encoders.')
            stages.append(
                OneHotEncoder(
                    inputCols=[f'{col}_numeric' for col in string_columns],
                    outputCols=[f'{col}_vector' for col in string_columns],
                )
            )
        # logger.debug('Skipping OHE.')
        logger.debug('Adding VectorAssembler.')
        stages.append(
            VectorAssembler(
                inputCols=(
                        [f'{col}_vector' for col in string_columns] +
                        numeric_columns
                ),
                outputCol='unscaled_' + config.Column.features_column,
                handleInvalid=handleInvalid
            )
        )
        logger.debug('Adding StandardScaler.')
        stages.append(
            StandardScaler(
                inputCol='unscaled_' + config.Column.features_column,
                outputCol=config.Column.features_column
            )
        )
        
        df = self._convert_numeric_null_to_zero(df)
        
        logger.debug('Fitting assembler.')
        self.assembler: Pipeline = Pipeline(stages=stages).fit(df)

    def _optional_smelting(
            self,
            df: config.Types.DataFrame,
            counterfactual_mode: bool = False,
            **kwargs
    ) -> config.Types.DataFrame:
        """
        Optional method to run a smelter pipeline on the DataFrame.

        Returns:
            config.Types.DataFrame: The DataFrame after running the smelter pipeline.
        """
        logger.debug('No optional smelting to run.')
        logger.debug('Counterfactual mode: %s', counterfactual_mode)
        return df

    def _optional_adstocking(
            self,
            df: config.Types.DataFrame
    ) -> config.Types.DataFrame:
        """
        Optional method to run adstocking on the DataFrame.

        Returns:
            config.Types.DataFrame: The DataFrame after running adstocking.
        """
        if self.model_config.adstock_params is None:
            logger.debug('No optional adstocking to run.')
            return df

        logger.debug('Running adstocking on DataFrame.')
        return reduce(
            lambda sdf, adstock_column: (
                sdf
                .transform(
                    self.adstocker.run,
                    adstock_column=adstock_column
                )
            ),
            self.model_config.adstock_params.keys(),
            df
        )

    def run(
            self,
            ignore_start_date: bool = True,
            counterfactual_mode: bool = False,
            **kwargs
    ) -> config.Types.DataFrame:
        """
        Transforms the DataFrame using the VectorAssembler.

        Parameters:
            ignore_start_date (bool): Whether or not to ignore the start_date defined in `self.model_config.` In some
            cases - inference should use data from the begininning of time.

        Returns:
            config.Types.DataFrame: The transformed DataFrame.
        """

        if self.df_:
            return (
                self.df_
                .transform(
                    self._filter_study_period,
                    start_date=None if ignore_start_date else self.model_config.start_date,
                    end_date=self.model_config.end_date,
                )
            )

        _df: config.Types.DataFrame = (
            self.df
            .transform(self._filter_model_triangle)
            .transform(  # end date can be filtered now - start date filtering depends on training or inference
                self._filter_study_period,
                start_date=None,
                end_date=self.model_config.end_date,
            )
            # .cache()
            .transform(
                self._optional_smelting,
                counterfactual_mode=counterfactual_mode,
                **kwargs
            )
            .transform(self._optional_adstocking)
        )

        if self.assembler:
            logger.debug('Found existing VectorAssembler.')
        if not self.assembler:
            logger.debug('Creating (or getting) VectorAssembler.')
            self._make_vector_assembler(_df)

        logger.debug('Transforming DataFrame using VectorAssembler.')
        # self.df_: config.Types.DataFrame = self.assembler.transform(_df).checkpoint()
        self.df_: config.Types.DataFrame = self.assembler.transform(_df)
        return (
            self.df_
            .transform(
                self._filter_study_period,
                start_date=None if ignore_start_date else self.model_config.start_date,
                end_date=self.model_config.end_date,
            )
        )

    def get_engineer_metadata(self, ignore_start_date: bool = True, counterfactual_mode: bool = False, **kwargs):
        _df: config.Types.DataFrame = (
            self.df
            .transform(self._filter_model_triangle)
            .transform(  # end date can be filtered now - start date filtering depends on training or inference
                self._filter_study_period,
                start_date=None,
                end_date=self.model_config.end_date,
            )
            # .cache()
            .transform(
                self._optional_smelting,
                counterfactual_mode=counterfactual_mode,
                **kwargs
            )
            .transform(self._optional_adstocking)
        )

        if self.assembler:
            logger.debug('Found existing VectorAssembler.')
        if not self.assembler:
            logger.debug('Creating (or getting) VectorAssembler.')
            self._make_vector_assembler(_df)
        return self.assembler


class FeatureEngineerValidator:
    """
    A class used to validate feature engineering configurations for machine learning models.

    Attributes
    ----------
    MODEL_TYPE : str
        The type of the model to be validated.
    feature_column_patterns : Sequence[str]
        A list of patterns for feature columns to be validated.

    Methods
    -------
    run() -> None
        Executes the validation checks for model type and feature column patterns.
    validate() -> bool
        Checks if any of the feature column patterns match the model type.
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        assert not args, 'No positional arguments are allowed.'
        self.model_type: str = kwargs.pop('model_type')
        self.feature_column_patterns: Sequence[str] = kwargs.pop('feature_column_patterns')

        if len(kwargs) > 0:
            logger.warning('Received unexpected keyword arguments: %s', kwargs)
            pass

    def run(self) -> None:
        """
        Executes the run method which validates the model type and feature column patterns.

        This method performs the following actions:
        - Validates the model type using the __validate_model_type method.
        - Validates the feature column patterns using the __validate_feature_column_patterns method.

        Raises:
            AssertionError: If any of the validations fail.
        """
        logger.debug('Validating FeatureEngineer configuration.')
        assert all([
            self.__validate_model_type(),
            self.__validate_feature_column_patterns()
        ])

    def __validate_model_type(self) -> bool:
        """
        Validates that the model type is within the allowed models specified in the configuration.

        Returns:
            bool: True if the model type is valid.

        Raises:
            AssertionError: If the model type is not in the allowed models list.
        """

        assert self.model_type in config.ml.ALLOWED_MODEL_TYPES, (
            'Invalid MODEL_TYPE. Received "{}" but expected one of: {}'.format(
                self.model_type,
                config.ml.ALLOWED_MODEL_TYPES
            )
        )
        return True

    def __validate_feature_column_patterns(self) -> bool:
        """
        Validates that each pattern in feature_column_patterns is allowed.

        Returns:
            bool: True if all patterns are valid.

        Raises:
            AssertionError: If any pattern in feature_column_patterns is not allowed.
        """
        for _pattern in self.feature_column_patterns:
            # FIXME these patterns are too strict - we need to check that one regex is a subset of the the other, not
            # that the patterns match exactly.

            pass
            # assert _pattern in config.ml.ALLOWED_FEATURE_PATTERNS, (
            #     'Invalid feature_column_patterns. Received "{}" but expected one of: {}'.format(
            #         self.FEATURE_COLUMN_PATTERNS,
            #         config.ml.ALLOWED_FEATURE_PATTERNS
            #     )
            # )
        return True

    def validate(self) -> bool:
        """
        Validates the model type against allowed feature patterns.

        Returns:
            bool: True if the model type matches any of the allowed feature patterns, False otherwise.
        """

        return any(map(
            lambda pattern: re.match(pattern, self.model_type),
            config.ml.ALLOWED_FEATURE_PATTERNS
        ))


## Path Type Feature Engineer Bases

class PathAEngineer(FeatureEngineer):
    ADSTOCKER: object = PathAAdstock

    def _maybe_dedupe_npp(
            self,
            df: config.Types.DataFrame
    ) -> config.Types.DataFrame:
        suggestion_type: str = self.model_config.suggestion_type
        logger.debug('Maybe de-duping NPP suggestions and HO emails. Suggestion type: %s', suggestion_type)
        return (
            PathAAggregator(
                data=df,
                suggestion_type=suggestion_type
            )
            .run()
        )

    def _optional_smelting(
            self,
            df: config.Types.DataFrame,
            **kwargs
    ) -> config.Types.DataFrame:
        logger.debug('Running optional smelting for Path A models.')
        return (
            df
            .transform(self._maybe_dedupe_npp)
        )


class PathBEngineer(FeatureEngineer):
    ADSTOCKER: object = PathBAdstock

    def _aggregate_to_hcp_week(
            self,
            df: config.Types.DataFrame
    ) -> config.Types.DataFrame:
        suggestion_type: str = self.model_config.suggestion_type
        logger.debug('Collapsing rep_id dimension for Path B models for suggestion type: %s', suggestion_type)
        return (
            PathBAggregator(
                data=df,
                suggestion_type=suggestion_type
            )
            .run()
        )


## Treatment Effect Engineer Base

class TreatmentEffectEngineer(FeatureEngineer):

    @property
    def spark_session(self) -> SparkSession:
        return self.df.sparkSession

    def _join_propensity_score_predictions(
            self,
            df: config.Types.DataFrame
    ) -> config.Types.DataFrame:
        # TODO Move this to a Smelter.

        # propensity_model: str = self.MODEL_TYPE.replace(
        #     'TreatmentEffect',
        #     'Propensity'
        # )
        #
        # propensity_data_path: PurePath = (
        #     self.model_config.path.data_write_path(
        #         model_type=propensity_model,
        #         model_id=self.model_config.dependent_model_id,
        #     )
        # )
        #
        # logger.debug(
        #     'Adding propensity scores from %s to model matrix for %s Model: %s',
        #     propensity_model,
        #     self.MODEL_TYPE,
        #     propensity_data_path
        # )
        #
        # propensity_df: config.Types.DataFrame = (
        #     self.spark_session
        #     .read
        #     .parquet(
        #         str(propensity_data_path)
        #     )
        # )
        propensity_df = self.prop_pred_df
        # logger.debug("The propensity df has %d rows.", propensity_df.count())

        propensity_columns: Sequence[str] = list(
            set([
                "rep_id",
                "abbott_customer_id",
                "week_end_date",
                self.model_config.propensity_column
            ])
            .intersection(set(propensity_df.columns))
        )
        join_columns: Sequence[str] = list(
            set([
                "rep_id",
                "abbott_customer_id",
                "week_end_date"
            ])
            .intersection(set(propensity_df.columns))
        )
        logger.debug(
            'Using propensity columns: %s and join columns: %s',
            propensity_columns,
            join_columns
        )

        df_w_propensity = df.join(
            propensity_df.select(
                *propensity_columns
            ),
            on=join_columns,
            how="left",
        )

        # logger.debug('The combined dataframe has %d rows', df_w_propensity.count())

        # assert df_w_propensity.count() == df.count(), (
        #     "Error adding propensity scores - index mismatch between propensity data and raw data."
        # )

        # logger.debug(
        #     'The average propensity is %.4f', (
        #         df_w_propensity
        #         .select(self.model_config.propensity_column)
        #         .agg({self.model_config.propensity_column: 'avg'})
        #         .collect()
        #         [0][0]
        #     )
        # )

        return df_w_propensity


## Propensity Models

class PathAPropensityEngineer(PathAEngineer):
    r"""
    A feature engineer class for Path A Propensity Model.

    Attributes:
        MODEL_TYPE (str): A string representing the type of model, set to 'PathA_PropensityModel'.
        FEATURE_COLUMN_PATTERNS (Sequence[str]): A list of regex patterns used to identify relevant feature columns
    """

    MODEL_TYPE: str = 'PathA_PropensityModel'
    FEATURE_COLUMN_PATTERNS: Sequence[str] = [
        r'^confound_path_[Aa]_.*'
    ]


class PathBPropensityEngineer(PathBEngineer):
    r"""
    A feature engineer class for Path B Propensity Model.

    Attributes:
        MODEL_TYPE (str): The type of the model, set to 'PathB_PropensityModel'.
        FEATURE_COLUMN_PATTERNS (Sequence[str]): A list of regex patterns to match feature column names.
    """

    MODEL_TYPE: str = 'PathB_PropensityModel'
    FEATURE_COLUMN_PATTERNS: Sequence[str] = [
        r'^CONFOUND_PATH_[Bb]_.*'
    ]

    def _optional_smelting(
            self,
            df: config.Types.DataFrame,
            **kwargs
    ) -> config.Types.DataFrame:
        logger.debug('Running optional smelting for Path B Propensity Model.')
        return (
            df
            .transform(self._aggregate_to_hcp_week)
        )


## Treatment Effect Models
class PathATreatmentEffectEngineer(
    PathAEngineer,
    TreatmentEffectEngineer
):
    r"""
    A feature engineer class for Path A treatment effect models.

    Attributes:
        MODEL_TYPE (str): Specifies the type of model as 'PathA_TreatmentEffectModel'.
        FEATURE_COLUMN_PATTERNS (Sequence[str]): A list of regex patterns to identify relevant feature columns.
    """

    MODEL_TYPE: str = 'PathA_TreatmentEffectModel'
    FEATURE_COLUMN_PATTERNS: Sequence[str] = [
        r'^covariate_path_[Aa]_(?!.*_triggered_emails_ho_npp$).*',
        r'^propensity_score_path_[Aa]_suggestion_count$',
        r'^percent_other_hcps_suggested(_adstock)?$',
        r'^suggestion_count(_adstock)?$',
        # r'^.*_global_index$',
    ]

    def _maybe_add_actual_percent_other_hcps_suggested(
            self,
            df: config.Types.DataFrame,
            counterfactual_mode: bool = False,
            **kwargs
    ) -> config.Types.DataFrame:
        if not counterfactual_mode:
            logger.debug('Not running in counterfactual mode so not adding actual percent other hcps suggested.')
            return df

        logger.debug('Adding actual percent other hcps suggested because running in counterfactual mode.')
        columns_to_drop: Sequence[str] = [
            _col
            for _col in df.columns
            if re.match(r'^percent_other_hcps_suggested(_adstock)?$', _col)
        ]
        assert 'actual_percent_other_hcps_suggested' in kwargs, (
            'Cannot add actual_percent_other_hcps_suggested because '
            f'it was not passed as a kwarg: {kwargs}'
        )
        logger.debug('Dropping columns: %s', columns_to_drop)
        return (
            df
            .drop(*columns_to_drop)
            .join(
                kwargs['actual_percent_other_hcps_suggested'],
                on=[
                    'week_end_date',
                    'abbott_customer_id',
                    'rep_id'
                ],
                how='left'
            )
        )

    def _optional_smelting(
            self,
            df: config.Types.DataFrame,
            counterfactual_mode: bool = False,
            **kwargs
    ) -> config.Types.DataFrame:
        activity_index_smelter: GlobalActivityIndexSmelter = GlobalActivityIndexSmelter()
        rep_activity_index_smelter: RepActivityIndexSmelter = RepActivityIndexSmelter()
        hcp_activity_index_smelter: HCPActivityIndexSmelter = HCPActivityIndexSmelter()

        logger.debug('Running optional smelting for Path A Treatment Effect Model.')
        return (
            super()._optional_smelting(df)  # Calls PathAEngineer._optional_smelting
            .transform(self._join_propensity_score_predictions)
            .transform(PercentOtherHCPSSuggestedSmelter().transform)
            # .transform(
            #     self._maybe_add_actual_percent_other_hcps_suggested,
            #     counterfactual_mode=counterfactual_mode,
            #     **kwargs
            # )
            # .transform(
            #     rep_activity_index_smelter.transform,
            #     activity_to_index='action_count'
            # )
            # .transform(
            #     hcp_activity_index_smelter.transform,
            #     activity_to_index='action_count'
            # )
            .transform(
                activity_index_smelter.transform,
                activity_to_index='action_count'
            )
        )


class PathBTreatmentEffectEngineer(
    PathBEngineer,
    TreatmentEffectEngineer,
):
    r"""
    A feature engineer class for Path B treatment effect models.

    Attributes:
        MODEL_TYPE (str): The type of model this engineer is designed for.
        FEATURE_COLUMN_PATTERNS (Sequence[str]): A list of regex patterns to identify relevant feature columns.
    """

    MODEL_TYPE: str = 'PathB_TreatmentEffectModel'
    FEATURE_COLUMN_PATTERNS: Sequence[str] = [
        r'^covariate_path_[Bb]_.*',
        r'^propensity_score_path_[Bb]_action_count$',
        r'^action_count(_adstock)?$',
        r'^suggestion_count(_adstock)?$',
        # r'^.*_global_index$',
    ]

    def _optional_smelting(
            self,
            df: config.Types.DataFrame,
            **kwargs
    ) -> config.Types.DataFrame:
        activity_index_smelter: GlobalActivityIndexSmelter = GlobalActivityIndexSmelter()

        logger.debug('Running optional smelting for Path B Treatment Effect model.')
        return (
            df
            .transform(self._aggregate_to_hcp_week)
            .transform(self._join_propensity_score_predictions)
            # .transform(
            #     activity_index_smelter.transform,
            #     activity_to_index='action_count'
            # )
            # .transform(
            #     activity_index_smelter.transform,
            #     activity_to_index='outcome_count'
            # )
        )