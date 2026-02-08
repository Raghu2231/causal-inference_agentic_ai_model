'''
1. Set the two suggestion features (‘suggestion’ and ‘prop other hcps with suggestion’) to zero.
2. Generate predictions from the suggestion -> call model.
3. Substitute predicted E[call] for actual calls, and plug these into the call -> XRx model.
4. Repeat steps 2-3, but on a dataset where the suggestion features are left alone (i.e. skip (1))
5. Calculate overall impact of suggestions as avg[ (4) - (3) ] .

'''

import json
import pickle
import subprocess
from uuid import uuid4
from itertools import product
from functools import reduce
from pathlib import Path, PurePath
from typing import Dict, Optional, Sequence, Tuple, List
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F, DataFrame
import mlops_suggestion_measurement.suggestions_measurement as sm
from mlops_suggestion_measurement.suggestions_measurement.logging import logger
from mlops_suggestion_measurement.suggestions_measurement import config
from mlops_suggestion_measurement.suggestions_measurement.utils.orchestrator import get_master_input, get_engineers, \
    get_alchemists
from mlops_suggestion_measurement.suggestions_measurement.utils.archivist import Archivist
from ..spark import PathAAggregator


class Refinery:

    def __init__(
            self,
            alchemists: Sequence[object],
            spark_session: config.Types.SparkSession,
            theme_run: bool = False
    ):
        self.alchemists = alchemists
        self.id = str(uuid4())
        self.spark_session = spark_session
        self.theme_run: bool = theme_run

        self.incremental_action: Optional[config.Types.DataFrame] = None
        self.incremental_outcome: Optional[config.Types.DataFrame] = None
        self.theme_incremental_action: Optional[Dict[str, config.Types.DataFrame]] = None
        self.theme_incremental_outcome: Optional[Dict[str, config.Types.DataFrame]] = None
        self.counterfactual_action_counts: Optional[config.Types.DataFrame] = None
        self.predicted_action_counts: Optional[config.Types.DataFrame] = None
        self.counterfactual_outcome_counts: Optional[Dict[str, config.Types.DataFrame]] = dict()
        self.predicted_outcome_counts: Optional[Dict[str, config.Types.DataFrame]] = dict()
        self.theme_predicted_action_counts: Optional[Dict[Tuple[str, str], config.Types.DataFrame]] = dict()
        self.theme_counterfactual_outcome_counts: Optional[Dict[Tuple[str, str], config.Types.DataFrame]] = dict()
        self.theme_predicted_outcome_counts: Optional[Dict[Tuple[str, str], config.Types.DataFrame]] = dict()

        self.start_date = self.alchemists["model_config"].start_date
        self.end_date = self.alchemists["model_config"].end_date

        logger.debug('Initialized refinery with ID: %s', self.id)

    def __getstate__(self):
        """
        Custom serialization: remove non-pickleable attributes like the Spark session.
        """
        logger.debug(f"__getstate__ called for Refinery with ID {self.id}")
        state = self.__dict__.copy()
        logger.debug(f"State keys: {state.keys()}")

        excluded_keys = [
            "spark_session",
            "incremental_action",
            "incremental_outcome",
            "predicted_action_counts",
            "counterfactual_action_counts",
            "counterfactual_outcome_counts",
            "predicted_outcome_counts",
            "alchemists",
            "_incremental_action",
            "_incremental_outcome",
            "_theme_incremental_action",
            "_theme_incremental_outcome",
            "theme_incremental_action",
            "theme_incremental_outcome",
            "theme_predicted_outcome_counts",
            "theme_predicted_action_counts",
            "theme_counterfactual_outcome_counts",
        ]

        for key in excluded_keys:
            if key in state:
                logger.debug(f"Excluding key: {key}")
                del state[key]

        logger.debug("Refinery object with ID %s is being pickled.", self.id)
        return state

    def __setstate__(self, state):
        """
        Restores the Refinery object state after unpickling by reinitializing any excluded or non-pickleable attributes.
        """
        self.__dict__.update(state)
        logger.debug(f"__setstate__ called for Refinery with ID {self.id}")

        session_manager = sm.utils.spark.SparkSessionManager(
            {
                "spark.executor.cores": "1",
                "spark.executor.instances": "60",
            }
        )
        session = session_manager.start_session()
        self.spark_session = session

        self.incremental_action = None
        self.incremental_outcome = None
        self.predicted_action_counts = None
        self.counterfactual_action_counts = None
        self.counterfactual_outcome_counts = None
        self.predicted_outcome_counts = None
        self.alchemists = {}

        logger.debug(f"Refinery object with ID {self.id} has been unpickled and reinitialized.")

    @property
    def refinery_write_path(self) -> str:
        return str(self.alchemists['PathA_PropensityModel'].model_config.path.refinery_write_path())

    def _save_refinery(self) -> config.Types.Self:
        # Create local temporary file
        local_dir = Path("temp/refinery")
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / "refinery.pkl"

        logger.debug('Saving refinery to local file: %s', local_file)
        with open(local_file, "wb") as file:
            pickle.dump(self, file)

        # Define HDFS path
        hdfs_path: PurePath = self.refinery_write_path

        # Use subprocess to copy to HDFS
        subprocess.run(["hdfs", "dfs", "-put", "-f", str(local_file), hdfs_path], check=True)
        return self

    @staticmethod
    def _zero_out_suggestions(
            sdf: config.Types.DataFrame
    ) -> config.Types.DataFrame:

        suggestion_columns: Sequence[str] = list(filter(
            lambda _column: 'suggest' in _column and _column != 'suggestion_type',
            sdf.columns
        ))
        logger.debug('Zeroing out suggestion features: %s', suggestion_columns)
        return reduce(
            lambda _sdf, _col_umn: _sdf.withColumn(
                _col_umn,
                F.lit(0)
            ),
            suggestion_columns,
            sdf
        )
        # return (
        #     sdf
        #     .withColumn(
        #         'suggestion_count',
        #         F.lit(0)
        #     )
        #     .withColumn(
        #         'suggestion_count_adstock',
        #         F.lit(0)
        #     )
        #     # TODO zero-out the themes as well
        # )

    def _get_themes(
            self
    ) -> config.Types.Self:
        if self.theme_run:
            logger.debug('Getting themes.')
            if self.alchemists["model_config"].suggestion_type == 'npp':
                self.themes: Sequence[str] = list(filter(
                    lambda _column: 'theme' in _column and 'triggered_emails_ho_np' in _column,
                    (
                        self.alchemists["feature_engineer_input"]
                        .columns
                    )
                ))
                logger.debug('Found themes: %s', self.themes)
                return self

            self.themes: Sequence[str] = list(filter(
                lambda _column: (
                        'theme' in _column and 'qualified' in _column
                    # 'theme' in _column and 'qualified' not in _column
                    # and 'triggered_emails_ho_np' not in _column
                ),
                (
                    self.alchemists["feature_engineer_input"]
                    .columns
                )
            ))
            logger.debug('Found themes: %s', self.themes)
            return self

        logger.debug('No need to get themes because not running in theme mode.')
        return self

    def _get_actual_percent_other_hcps_suggested(
            self,
    ) -> config.Types.DataFrame:

        logger.debug('Getting actual percent other hcps suggested.')

        return (
            self.alchemists
            ['PathA_TreatmentEffectModel']
            .feature_engineer
            .df_
            .select(*[
                'abbott_customer_id',
                'week_end_date',
                'rep_id',
                'percent_other_hcps_suggested'
            ])
        )

    def _get_counterfactual_action_counts(
            self
    ) -> config.Types.Self:

        logger.debug('Getting counterfactual action counts.')
        '''
        original_data = self.alchemists["PathA_TreatmentEffectModel"].feature_engineer.df
        original_model_config = self.alchemists["PathA_TreatmentEffectModel"].feature_engineer.model_config

        modified_data = original_data.transform(self._zero_out_suggestions)
        # actual_percent_other_hcps_suggested = self._get_actual_percent_other_hcps_suggested()

        engineer = sm.utils.ml.engineer.PathATreatmentEffectEngineer(
            df=modified_data,
            model_config=original_model_config
        )
        logger.debug('Getting assembler and feature_columns from PathA_TreatmentEffectModel.')
        engineer.assembler: PipelineModel = (
            self
            .alchemists["PathA_TreatmentEffectModel"]
            .feature_engineer
            .assembler
        )
        engineer.feature_columns: Sequence[str] = (
            self
            .alchemists["PathA_TreatmentEffectModel"]
            .feature_engineer
            .feature_columns
        )
        '''

        self.counterfactual_action_counts: config.Types.DataFrame = (
            # engineer.run(
            #     ignore_start_date=True,
            #     counterfactual_mode=True,
            #     # actual_percent_other_hcps_suggested=actual_percent_other_hcps_suggested
            # )
            self.alchemists["counterfactual_table"]
            .transform(self.alchemists['trained_model'].transform)
            .transform(lambda sdf: (
                sdf
                .select([
                    "abbott_customer_id",
                    "rep_id",
                    "week_end_date",
                    *[
                        F.col(_col).alias(f'{_col}_counterfactual')

                        for _col in filter(
                            lambda _column: _column in sdf.columns,
                            [
                                'suggestion_count',
                                'suggestion_count_adstock',
                                'percent_other_hcps_suggested'
                            ]
                        )
                    ],
                    (
                        F.when(
                            F.col("prediction") < 0,
                            0
                        )
                        .otherwise(
                            F.col('prediction')
                        )
                        .alias("counterfactual_action_count")
                    ),
                ])
            ))
            # .checkpoint()
        )

        return self

    def _get_predicted_action_counts(
            self,
            theme: Optional[str] = None
    ) -> config.Types.Self:
        if not theme:
            logger.debug('Getting predicted action counts.')
            self.predicted_action_counts: config.Types.DataFrame = (
                self.alchemists['prediction_data']
                .transform(lambda sdf: (
                    sdf
                    .select([
                        "abbott_customer_id",
                        "rep_id",
                        "week_end_date",
                        *[
                            F.col(_col).alias(f'{_col}_factual')

                            for _col in filter(
                                lambda _column: _column in sdf.columns,
                                [
                                    'suggestion_count',
                                    'suggestion_count_adstock',
                                    'percent_other_hcps_suggested',
                                    'action_count',
                                ]
                            )
                        ],
                        (
                            F.when(
                                F.col("prediction") < 0,
                                0
                            )
                            .otherwise(
                                F.col('prediction')
                            )
                            .alias("predicted_action_count")
                        ),
                        (
                            F.col(
                                self.alchemists['model_config']
                                .propensity_column
                            )
                            .alias('propensity')
                        )
                    ])
                ))
                # .checkpoint()
            )

        if theme:
            logger.debug('Getting predicted action counts for theme: %s', theme)
            '''
            original_data = self.alchemists["PathA_TreatmentEffectModel"].feature_engineer.df
            original_model_config = self.alchemists["PathA_TreatmentEffectModel"].feature_engineer.model_config
            modified_data = (
                original_data
                .withColumnRenamed(
                    'suggestion_count',
                    'total_suggestion_count'
                )
                .withColumn(
                    'suggestion_count',
                    F.when(
                        F.col('total_suggestion_count') > 0,
                        F.col(theme)
                    ).otherwise(0)
                )
                # .drop(*[
                #     theme
                # ])
            )

            engineer = sm.utils.ml.engineer.PathATreatmentEffectEngineer(
                df=modified_data,
                model_config=original_model_config
            )
            logger.debug('Getting assembler and feature_columns from PathA_TreatmentEffectModel.')
            engineer.assembler: PipelineModel = (
                self
                .alchemists["PathA_TreatmentEffectModel"]
                .feature_engineer
                .assembler
            )
            engineer.feature_columns: Sequence[str] = (
                self
                .alchemists["PathA_TreatmentEffectModel"]
                .feature_engineer
                .feature_columns
            )
            '''
            self.theme_predicted_action_counts[theme]: config.Types.DataFrame = (
                # engineer.run(
                #     ignore_start_date=True,
                #     counterfactual_mode=False,
                # )
                self.alchemists["theme_df"].get(theme, None)
                .transform(self.alchemists['trained_model'].transform)
                .transform(lambda sdf: (
                    sdf
                    .select([
                        "abbott_customer_id",
                        "rep_id",
                        "week_end_date",
                        *[
                            F.col(_col).alias(f'{_col}_factual')

                            for _col in filter(
                                lambda _column: _column in sdf.columns,
                                [
                                    'suggestion_count',
                                    'suggestion_count_adstock',
                                    'percent_other_hcps_suggested',
                                    'action_count',
                                ]
                            )
                        ],
                        (
                            F.when(
                                F.col("prediction") < 0,
                                0
                            )
                            .otherwise(
                                F.col('prediction')
                            )
                            .alias("predicted_action_count")
                        ),
                        (
                            F.col(
                                self.alchemists['model_config']
                                .propensity_column
                            )
                            .alias('propensity')
                        )
                    ])
                ))
                # .checkpoint()
            )
        return self

    def _get_theme_predicted_action_counts(
            self,
    ) -> config.Types.Self:
        if self.theme_run:
            logger.debug('Getting theme predicted action counts.')
            for _theme in self.themes:
                logger.debug('Running for theme: %s', _theme)
                self._get_predicted_action_counts(theme=_theme)
            return self

        logger.debug('No need to get theme predicted action counts because not running in theme mode.')
        return self

    def _get_counterfactual_outcome_counts(
            self,
            outcome_type: str,
            theme: Optional[str] = None,
    ) -> config.Types.Self:

        logger.debug('Getting counterfactual outcome counts.')
        if theme:
            logger.debug('Getting counterfactual outcome counts for theme: %s', theme)
        '''
        original_data = (
            self.alchemists
            [f"PathB_TreatmentEffectModel__{outcome_type}"]
            .feature_engineer
            .df
            .transform(self._zero_out_suggestions)
            .transform(lambda sdf: (
                PathAAggregator(
                    data=sdf,
                    suggestion_type=(
                        self.alchemists
                        [f"PathB_TreatmentEffectModel__{outcome_type}"]
                        .feature_engineer
                        .model_config
                        .suggestion_type
                    )
                )
                .run()
            ))
        )
        original_model_config = self.alchemists[f"PathB_TreatmentEffectModel__{outcome_type}"].feature_engineer.model_config

        original_action_count =  original_data.select(F.sum('action_count').alias('action_count')).toPandas()['action_count'].tolist()
        logger.debug("Dropping action_count, adding counterfactual_action_count, original action count: %s", original_action_count)
        modified_data = (
            original_data.drop("action_count")
            .join(
                self.counterfactual_action_counts,
                on=["rep_id", "abbott_customer_id", "week_end_date"],
                how='left'
            )
            .withColumnRenamed("counterfactual_action_count", "action_count")
        )
        modified_action_count = modified_data.select(F.sum('action_count').alias('action_count')).toPandas()['action_count'].tolist()
        logger.debug('Final action_count: %s', modified_action_count)

        engineer = sm.utils.ml.engineer.PathBTreatmentEffectEngineer(
            modified_data,
            original_model_config
        )
        logger.debug('Getting assembler and feature_columns from PathB_TreatmentEffectModel__%s.', outcome_type)
        engineer.assembler: PipelineModel = (
            self
            .alchemists[f"PathB_TreatmentEffectModel__{outcome_type}"]
            .feature_engineer
            .assembler
        )
        engineer.feature_columns: Sequence[str] = (
            self
            .alchemists[f"PathB_TreatmentEffectModel__{outcome_type}"]
            .feature_engineer
            .feature_columns
        )
        '''

        out: config.Types.DataFrame = (
            # engineer.run(ignore_start_date=True)
            self.alchemists[f"counter_factual_{outcome_type}_df"]
            .transform(self.alchemists[f'PathB_TreatmentEffectModel__{outcome_type}_trained_model'].transform)
            .select(*[
                'abbott_customer_id',
                'week_end_date',
                (
                    F.when(
                        F.col("prediction") < 0,
                        0
                    )
                    .otherwise(
                        F.col('prediction')
                    )
                    .alias("counterfactual_outcome_count")
                ),
            ])
            # .checkpoint()
        )

        if not theme:
            self.counterfactual_outcome_counts[outcome_type]: config.Types.DataFrame = out

        if theme:
            logger.debug('Setting counterfactual outcome counts for theme: %s', theme)
            self.theme_counterfactual_outcome_counts[(outcome_type, theme)]: config.Types.DataFrame = out
            logger.debug('Theme counterfactual outcome counts keys: %s',
                         self.theme_counterfactual_outcome_counts.keys())

        return self

    def _get_theme_counterfactual_outcome_counts(
            self,
    ) -> config.Types.Self:
        if self.theme_run:
            for _theme, _outcome_type in product(self.themes, ['nbrx', 'trx']):
                logger.debug('Getting theme counterfactual outcome counts: %s, %s', _theme, _outcome_type)
                self._get_counterfactual_outcome_counts(
                    theme=_theme,
                    outcome_type=_outcome_type
                )
            return self

        logger.debug('No need to get theme counterfactual outcome counts because not running in theme mode.')
        return self

    def _get_predicted_outcome_counts(
            self,
            outcome_type: str,
            theme: Optional[str] = None
    ) -> config.Types.Self:
        """ Get the predicted outcome count USING PREDICTED ACTION COUNT (not actual action count) """
        logger.debug('Getting predicted outcome counts.')
        '''
        original_data = (
            self.alchemists
            [f"PathB_TreatmentEffectModel__{outcome_type}"]
            .feature_engineer
            .df
            .transform(lambda sdf: (
                PathAAggregator(
                    data=sdf,
                    suggestion_type=(
                        self.alchemists
                        [f"PathB_TreatmentEffectModel__{outcome_type}"]
                        .feature_engineer
                        .model_config
                        .suggestion_type
                    )
                )
                .run()
            ))
        )
        original_model_config = self.alchemists[f"PathB_TreatmentEffectModel__{outcome_type}"].feature_engineer.model_config

        original_action_count =  original_data.select(F.sum('action_count').alias('action_count')).toPandas()['action_count'].tolist()
        logger.debug("Dropping action_count, adding predicted_action_count, original action count: %s", original_action_count)
        modified_data = (
            original_data.drop("action_count")
            .join(
                self.theme_predicted_action_counts
                .get(
                    theme,
                    self.predicted_action_counts
                ),
                on=["rep_id", "abbott_customer_id", "week_end_date"],
                how='left'
            )
            .withColumnRenamed("predicted_action_count", "action_count")
            .transform(lambda _sdf: (
                _sdf
                if not theme else
                _sdf
                .withColumnRenamed(
                    'suggestion_count',
                    'total_suggestion_count'
                )
                .withColumn(
                    'suggestion_count',
                    F.when(
                        F.col('total_suggestion_count') > 0,
                        F.col(theme)
                    ).otherwise(0)
                )
            ))
        )
        modified_action_count = modified_data.select(F.sum('action_count').alias('action_count')).toPandas()['action_count'].tolist()
        logger.debug('Final action_count: %s', modified_action_count)

        engineer = sm.utils.ml.engineer.PathBTreatmentEffectEngineer(
            modified_data,
            original_model_config,
        )
        logger.debug('Getting assembler and feature_columns from PathB_TreatmentEffectModel__%s.', outcome_type)
        engineer.assembler = (
            self
            .alchemists
            [f"PathB_TreatmentEffectModel__{outcome_type}"]
            .feature_engineer
            .assembler
        )
        engineer.feature_columns = (
            self
            .alchemists
            [f"PathB_TreatmentEffectModel__{outcome_type}"]
            .feature_engineer
            .feature_columns
        )
        '''
        if theme:
            df = self.alchemists["theme_predicted_outcome_counts"][outcome_type][theme]
        else:
            df = self.alchemists[f"predicted_outcome_{outcome_type}_df"]
            
        out: config.Types.DataFrame = (
            # engineer.run(ignore_start_date=True)
            df
            .transform(self.alchemists[f"PathB_TreatmentEffectModel__{outcome_type}_trained_model"].transform)
            .select(*[
                'week_end_date',
                'abbott_customer_id',
                (
                    F.when(
                        F.col("prediction") < 0,
                        0
                    )
                    .otherwise(
                        F.col('prediction')
                    )
                    .alias("predicted_outcome_count")
                ),
                (
                    F.col(
                        self.alchemists[f'model_config_{outcome_type}']
                        # .model_config
                        .propensity_column
                    )
                    .alias('propensity')
                )
            ])
            # .checkpoint()
        )

        if not theme:
            self.predicted_outcome_counts[outcome_type] = out
        if theme:
            logger.debug('Setting predicted outcome counts for theme: %s', theme)
            self.theme_predicted_outcome_counts[outcome_type, theme] = out
            logger.debug('Theme predicted outcome count keys: %s', self.theme_predicted_outcome_counts.keys())

        return self

    def _get_theme_predicted_outcome_counts(
            self,
    ) -> config.Types.Self:
        if self.theme_run:
            for _theme, _outcome_type in product(self.themes, ['nbrx', 'trx']):
                logger.debug('Getting theme predicted outcome counts: %s, %s', _theme, _outcome_type)
                self._get_predicted_outcome_counts(
                    theme=_theme,
                    outcome_type=_outcome_type
                )

            return self
        logger.debug('No need to get theme predicted outcome counts because not running in theme mode.')
        return self

    def _filter_study_period(
            self, df: config.Types.DataFrame,
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

        return _df

    def _get_incremental_action(
            self,
    ) -> config.Types.Self:
        logger.debug('Getting incremental actions.')
        self._incremental_action: config.Types.DataFrame = (
            reduce(
                DataFrame.unionByName,
                map(
                    lambda _outcome_type: (
                        self.predicted_action_counts
                        .join(
                            self.counterfactual_action_counts,
                            on=["rep_id", "abbott_customer_id", "week_end_date"]
                        )
                        # These predictions go back in time - we want to restrict to the study period
                        .transform(
                            self._filter_study_period,
                            start_date=self.start_date,
                            end_date=self.end_date,
                        )
                        # .withColumn(
                        #     # FIXME: this is brittle to running without ad stock
                        #     "predicted_action_count",
                        #     F.when(
                        #         F.col("suggestion_count_adstock_factual") == 0,
                        #         F.col("counterfactual_action_count")
                        #     )
                        #     .otherwise(F.col("predicted_action_count"))
                        # )
                        .withColumn(
                            'incremental_action',
                            F.col('predicted_action_count') - F.col('counterfactual_action_count')
                        )
                        .withColumn(
                            'outcome_type',
                            F.lit(_outcome_type)
                        )
                    ),
                    [
                        'nbrx',
                        'trx'
                    ]
                )
            )
        )

        if self.theme_run:
            logger.debug('Getting theme-based incremental actions.')
            self._theme_incremental_action: Dict[str, config.Types.DataFrame] = {
                theme: (
                    self.theme_predicted_action_counts[theme]
                    .join(
                        self.counterfactual_action_counts,
                        on=["rep_id", "abbott_customer_id", "week_end_date"]
                    )
                    .transform(
                        self._filter_study_period,
                        start_date=self.start_date,
                        end_date=self.end_date
                    )
                    # .withColumn(
                    #     # FIXME: this is brittle to running without ad stock
                    #     "predicted_action_count",
                    #     F.when(
                    #         F.col("suggestion_count_adstock_factual") == 0,
                    #         F.col("counterfactual_action_count")
                    #     )
                    #     .otherwise(F.col("predicted_action_count"))
                    # )
                    .withColumn(
                        'incremental_action',
                        F.col('predicted_action_count') - F.col('counterfactual_action_count')
                    )
                )
                for theme in self.theme_predicted_action_counts
            }

        return self

    def _get_incremental_outcome(
            self,
    ) -> config.Types.Self:
        logger.debug(f'Getting incremental outcomes.')

        # incremental_action_sugg_count = (
        #     self._incremental_action
        #     .groupby(
        #         'abbott_customer_id',
        #         'week_end_date',
        #         'outcome_type',
        #     )
        #     .agg(
        #         F.sum('suggestion_count_adstock_factual').alias('suggestion_count_adstock_factual')
        #     )
        # )

        self._incremental_outcome: config.Types.DataFrame = (
            reduce(
                DataFrame.unionByName,
                map(
                    lambda _outcome_type: (
                        self.predicted_outcome_counts[_outcome_type]
                        .join(
                            self.counterfactual_outcome_counts[_outcome_type],
                            on=["abbott_customer_id", "week_end_date"]
                        )
                        # These predictions go back in time - we want to restrict to the study period
                        .transform(
                            self._filter_study_period,
                            start_date=self.start_date,
                            end_date=self.end_date,
                        )
                        .withColumn(
                            'outcome_type',
                            F.lit(_outcome_type)
                        )
                        # .join(
                        #     incremental_action_sugg_count,
                        #     on=['abbott_customer_id', 'week_end_date', 'outcome_type'],
                        #     how='left'
                        # )
                        # .withColumn(
                        #     # FIXME: this is brittle to running without ad stock
                        #     "predicted_outcome_count",
                        #     F.when(
                        #         F.col("suggestion_count_adstock_factual") == 0,
                        #         F.col("counterfactual_outcome_count")
                        #     )
                        #     .otherwise(F.col("predicted_outcome_count"))
                        # )
                        .withColumn(
                            'incremental_outcome',
                            F.col('predicted_outcome_count') - F.col('counterfactual_outcome_count')
                        )
                        # .drop(
                        #     'suggestion_count_adstock_factual'
                        # )
                    ),
                    [
                        'nbrx',
                        'trx'
                    ]
                ),
            )
        )

        if self.theme_run:
            logger.debug('Getting theme-based incremental outcomes.')
            logger.debug('Counterfactual outcome themes: %s', self.theme_counterfactual_outcome_counts)
            logger.debug('Predicted outcome themes: %s', self.theme_predicted_outcome_counts)

            # theme_incremental_action_sugg_count = (
            #     self._theme_incremental_action
            #     .groupby(
            #         'abbott_customer_id',
            #         'week_end_date',
            #         'theme',
            #         'outcome_type',
            #     )
            #     .agg(
            #         F.sum('suggestion_count_adstock_factual').alias('suggestion_count_adstock_factual')
            #     )
            # )

            self._theme_incremental_outcome: Dict[str, config.Types.DataFrame] = {
                theme_outcome_type: (
                    self.theme_predicted_outcome_counts[theme_outcome_type]
                    .join(
                        self.theme_counterfactual_outcome_counts[theme_outcome_type],
                        on=["abbott_customer_id", "week_end_date"]
                    )
                    .transform(
                        self._filter_study_period,
                        start_date=self.start_date,
                        end_date=self.end_date
                    )
                    .withColumn(
                        'outcome_type',
                        F.lit(theme_outcome_type[0])
                    )
                    # .join(
                    #     theme_incremental_action_sugg_count,
                    #     on=['abbott_customer_id', 'week_end_date', 'theme', 'outcome_type'],
                    #     how='left'
                    # )
                    # .withColumn(
                    #     # FIXME: this is brittle to running without ad stock
                    #     "predicted_outcome_count",
                    #     F.when(
                    #         F.col("suggestion_count_adstock_factual") == 0,
                    #         F.col("counterfactual_outcome_count")
                    #     )
                    #     .otherwise(F.col("predicted_outcome_count"))
                    # )
                    .withColumn(
                        'incremental_outcome',
                        F.col('predicted_outcome_count') - F.col('counterfactual_outcome_count')
                    )
                    # .drop(
                    #     'suggestion_count_adstock_factual'
                    # )
                )
                for theme_outcome_type in self.theme_predicted_outcome_counts
            }

        return self

    def _get_flat_outputs(
            self, incremental_type
    ) -> config.Types.Self:
        logger.debug('Getting flat outputs.')
        if incremental_type == "action":
            self.incremental_action = (
                self._incremental_action
                .withColumn(
                    'theme',
                    F.lit('all')
                )
                .transform(lambda sdf: (
                    sdf
                    if not self.theme_run else
                    sdf
                    .unionByName(
                        reduce(
                            DataFrame.unionByName,
                            map(
                                lambda _theme: (
                                    self._theme_incremental_action[_theme]
                                    .withColumn(
                                        'theme',
                                        F.lit(_theme)
                                    )
                                ),
                                self.themes
                            )
                        )
                        .transform(lambda sdf: (
                            sdf
                            .withColumn(
                                'outcome_type',
                                F.lit('trx')
                            )
                            .unionByName(
                                sdf
                                .withColumn(
                                    'outcome_type',
                                    F.lit('nbrx')
                                )
                            )
                        ))
                    )
                ))
            )
        else:
            self.incremental_outcome = (
                self._incremental_outcome
                .withColumn(
                    'theme',
                    F.lit('all')
                )
                .transform(lambda sdf: (
                    sdf
                    if not self.theme_run else
                    sdf
                    .unionByName(
                        reduce(
                            DataFrame.unionByName,
                            map(
                                lambda _outcome_type_theme: (
                                    self._theme_incremental_outcome[_outcome_type_theme]
                                    .withColumn(
                                        'theme',
                                        F.lit(_outcome_type_theme[1])
                                    )
                                ),
                                product(['trx', 'nbrx'], self.themes)
                            )
                        )
                    )
                ))
            )
        return self

    def run(
            self
    ) -> config.Types.Self:
        logger.debug('Running refinery.')
        return (
            self
            ._get_themes()
            ._get_counterfactual_action_counts()
            ._get_predicted_action_counts()
            ._get_predicted_outcome_counts(
                outcome_type='nbrx'
            )
            ._get_predicted_outcome_counts(
                outcome_type='trx'
            )
            ._get_counterfactual_outcome_counts(
                outcome_type='nbrx'
            )
            ._get_counterfactual_outcome_counts(
                outcome_type='trx'
            )
            ._get_theme_predicted_action_counts()
            ._get_theme_counterfactual_outcome_counts()
            ._get_theme_predicted_outcome_counts()
            ._get_incremental_action()
            ._get_incremental_outcome()
            ._get_flat_outputs()
            ._save_refinery()
        )

    def path_a_run(
            self
    ) -> config.Types.Self:
        return (
            self
            ._get_themes()
            ._get_counterfactual_action_counts()
            ._get_predicted_action_counts()
            ._get_theme_predicted_action_counts()
            ._get_incremental_action()
            ._get_flat_outputs("action")
        )

    def path_b_run(
            self
    ) -> config.Types.Self:
        return (
            self
            ._get_themes()
            ._get_predicted_outcome_counts(
                outcome_type='nbrx'
            )
            ._get_predicted_outcome_counts(
                outcome_type='trx'
            )
            ._get_counterfactual_outcome_counts(
                outcome_type='nbrx'
            )
            ._get_counterfactual_outcome_counts(
                outcome_type='trx'
            )
            ._get_theme_counterfactual_outcome_counts()
            ._get_theme_predicted_outcome_counts()
            ._get_incremental_outcome()
            ._get_flat_outputs("outcome")
        )

    def summarize(self, ):
        assert self.incremental_outcome and self.incremental_action, \
            "No incremental outcome calculated - have you called refinery.run()?"

        incremental_outcome = (
            self.incremental_outcome
            .groupBy("theme")
            .agg(F.sum("incremental_outcome").alias("incremental_outcome"))
            .toPandas()
        )

        incremental_action = (
            self.incremental_action
            .groupBy("theme")
            .agg(F.sum("incremental_action").alias("incremental_action"))
            .toPandas()
        )

        return (
            incremental_outcome
            .merge(
                incremental_action,
                on="theme",
                how="outer"
            )
        )

        return summary


class RefineryLoader:
    def __init__(
            self,
            run_id: str,
            session,
            theme_run: bool = True,
            model_config_path: Optional[str] = None,
            refinery_path: Optional[str] = None
    ):
        self.run_id = run_id
        self.session = session
        self.model_config_path = (
            model_config_path
            if model_config_path is not None
            else f"suggestions_measurement__output/{self.run_id}/{self.run_id}/model_config/{{model_name}}/model_config.json"
        )
        self.refinery_path = (
            refinery_path
            if refinery_path is not None
            else f"suggestions_measurement__output/{self.run_id}/{self.run_id}/refinery/refinery.pkl"
        )
        Archivist().pull_and_unzip(run_id=self.run_id)
        self.model_configs = self._load_model_config()

    def _run_log_command(self, command: str) -> None:
        """
        Runs an HDFS CLI command and returns the output as a list of lines.

        :param command: A string command to execute
        """
        parsed_command: List[str] = command.strip().split(' ')
        logger.debug('Running command: %s', parsed_command)

        result: int = subprocess.call(parsed_command)

        try:
            result = subprocess.run(parsed_command, capture_output=True, text=True, check=True)
            return result.stdout.split("\n")
        except subprocess.CalledProcessError as e:
            logger.error("Command failed: %s", e)
            raise RuntimeError(f"Failed to execute command: {parsed_command} (Return code: {e.returncode})")

    def _load_model_config(self) -> Dict[str, object]:
        """
        Load and parse model configurations.
        """
        run_path = sm.config.PathConfig(
            run_id=self.run_id
        )
        model_names = [
            'PathA_PropensityModel',
            'PathA_TreatmentEffectModel',
            'PathB_PropensityModel',
            'PathB_TreatmentEffectModel__nbrx',
            'PathB_TreatmentEffectModel__trx',
        ]

        def load_single_model_config(
                model_name: str
        ) -> sm.config.ModelConfig:
            """
            Helper function to load a single model config.
            """
            model_config_path = self.model_config_path.format(model_name=model_name)
            logger.info(f"Loading model config for model {model_name} from {model_config_path}")
            # Read JSON from HDFS
            raw_json_rdd = self.session.sparkContext.textFile(model_config_path)
            raw_json = "\n".join(raw_json_rdd.collect())

            # Parse JSON and initialize ModelConfig
            config_data = json.loads(raw_json)
            model_config = sm.config.ModelConfig(**config_data)
            model_config.path = run_path
            logger.info(f"Successfully loaded config for {model_name}")
            return model_config

        model_configs_dict: Dict = {
            model_name: load_single_model_config(model_name)
            for model_name in model_names
        }
        return model_configs_dict

    def load_alchemists(self) -> Dict[str, object]:
        """
        Load and configure alchemists based on model configurations.
        """
        brand_ind: str = self.model_configs['PathA_PropensityModel'].brand_ind
        brand: str = {
            "Skyrizi PS/PSA": 'skyrizi',
            'Vraylar': "vraylar",
            'Simulated Data Brand': 'simdata',
        }.get(brand_ind)
        alloy: str = self.model_configs['PathA_PropensityModel'].alloy
        logger.debug(f"Creating alchemists for brand {brand} and alloy {alloy}")

        # Load master input and alchemists
        master_input: config.Types.DataFrame = get_master_input(
            brand=brand,
            alloy=alloy,
            session=self.session,
        )
        engineers: Dict[str, object] = get_engineers(
            master_input=master_input,
            model_configs=self.model_configs,
        )
        alchemists: Dict[str, object] = get_alchemists(
            engineers=engineers,
            model_configs=self.model_configs,
        )

        # Load models and data_with_predictions into alchemists
        for alchemist_name in alchemists.keys():
            if 'PathB_TreatmentEffectModel' in alchemist_name:
                model_path = f"suggestions_measurement__output/{self.run_id}/{self.run_id}/models/{alchemist_name.split('__')[0]}-{alchemists[alchemist_name].model_id}"
            else:
                model_path = f"suggestions_measurement__output/{self.run_id}/{self.run_id}/models/{alchemist_name}-{alchemists[alchemist_name].model_id}"

            pipeline_model = PipelineModel.load(model_path)
            alchemists[alchemist_name].trained_model = pipeline_model

            if 'propensity' in alchemist_name.lower():
                data_with_predictions_parquet_path = f"suggestions_measurement__output/{self.run_id}/{self.run_id}/data/{alchemist_name}-*"
                data_with_predictions_df = self.session.read.parquet(data_with_predictions_parquet_path)
                alchemists[alchemist_name].data_with_predictions = data_with_predictions_df

        return alchemists

    def load_refinery(self):
        """
        Load and attach alchemists and data to the refinery.
        """
        binary_rdd = self.session.sparkContext.binaryFiles(self.refinery_path)
        binary_content = binary_rdd.collect()[0][1]
        loaded_refinery = pickle.loads(binary_content)

        alchemists = self.load_alchemists()
        loaded_refinery.alchemists = alchemists

        incremental_action_parquet_path = f"suggestions_measurement__output/{self.run_id}/{self.run_id}/data/incremental_action"
        logger.debug('Loading incremental actions data from: %s', incremental_action_parquet_path)
        incremental_action_df = self.session.read.parquet(incremental_action_parquet_path)
        loaded_refinery.incremental_action = incremental_action_df

        incremental_outcome_parquet_path = f"suggestions_measurement__output/{self.run_id}/{self.run_id}/data/incremental_outcome"
        logger.debug('Loading incremental outcomes data from: %s', incremental_outcome_parquet_path)
        incremental_outcome_df = self.session.read.parquet(incremental_outcome_parquet_path)
        loaded_refinery.incremental_outcome = incremental_outcome_df

        predicted_action_counts_parquet_path = f"suggestions_measurement__output/{self.run_id}/{self.run_id}/data/predicted_action_counts"
        logger.debug('Loading predicted action counts data from: %s', predicted_action_counts_parquet_path)
        predicted_action_counts_df = self.session.read.parquet(predicted_action_counts_parquet_path)
        loaded_refinery.predicted_action_counts = predicted_action_counts_df

        counterfactual_action_counts_parquet_path = f"suggestions_measurement__output/{self.run_id}/{self.run_id}/data/counterfactual_action_counts"
        logger.debug('Loading counterfactual action counts data from: %s', counterfactual_action_counts_parquet_path)
        counterfactual_action_counts_df = self.session.read.parquet(counterfactual_action_counts_parquet_path)
        loaded_refinery.counterfactual_action_counts = counterfactual_action_counts_df

        predicted_outcome_counts_hdfs_path = f"suggestions_measurement__output/{self.run_id}/{self.run_id}/data/predicted_outcome_counts"
        logger.debug('Loading predicted outcome counts data from: %s', predicted_outcome_counts_hdfs_path)
        output_lines = self._run_log_command(
            f'hdfs dfs -ls {predicted_outcome_counts_hdfs_path}'
        )
        folder_names = [
            line.split()[-1].split('/')[-1] for line in output_lines
            if line and line.startswith("drwx")
        ]
        predicted_outcome_counts = {}
        for folder in folder_names:
            parquet_path = f"{predicted_outcome_counts_hdfs_path}/{folder}"
            predicted_outcome_counts[folder] = self.session.read.parquet(parquet_path)
        loaded_refinery.predicted_outcome_counts = predicted_outcome_counts

        counterfactual_outcome_counts_hdfs_path = f"suggestions_measurement__output/{self.run_id}/{self.run_id}/data/counterfactual_outcome_counts"
        logger.debug('Loading counterfactual outcome counts data from: %s', counterfactual_outcome_counts_hdfs_path)
        output_lines = self._run_log_command(
            f'hdfs dfs -ls {counterfactual_outcome_counts_hdfs_path}'
        )
        folder_names = [
            line.split()[-1].split('/')[-1] for line in output_lines
            if line and line.startswith("drwx")
        ]
        counterfactual_outcome_counts = {}
        for folder in folder_names:
            parquet_path = f"{counterfactual_outcome_counts_hdfs_path}/{folder}"
            counterfactual_outcome_counts[folder] = self.session.read.parquet(parquet_path)
        loaded_refinery.counterfactual_outcome_counts = counterfactual_outcome_counts

        if theme_run:
            theme_predicted_action_counts_hdfs_path = f"suggestions_measurement__output/{self.run_id}/{self.run_id}/data/theme_predicted_action_counts"
            logger.debug('Loading theme predicted action counts data from: %s', theme_predicted_action_counts_hdfs_path)
            output_lines = self._run_log_command(
                f'hdfs dfs -ls {theme_predicted_action_counts_hdfs_path}'
            )
            folder_names = [
                line.split()[-1].split('/')[-1] for line in output_lines
                if line and line.startswith("drwx")
            ]
            theme_predicted_action_counts = {}
            for folder in folder_names:
                parquet_path = f"{theme_predicted_action_counts_hdfs_path}/{folder}"
                theme_predicted_action_counts[folder] = self.session.read.parquet(parquet_path)
            loaded_refinery.theme_predicted_action_counts = theme_predicted_action_counts

            theme_predicted_outcome_counts_hdfs_path = f"suggestions_measurement__output/{self.run_id}/{self.run_id}/data/theme_predicted_outcome_counts"
            logger.debug('Loading theme predicted outcome counts data from: %s',
                         theme_predicted_outcome_counts_hdfs_path)
            output_lines = self._run_log_command(
                f'hdfs dfs -ls {theme_predicted_outcome_counts_hdfs_path}'
            )
            folder_names = [
                line.split()[-1].split('/')[-1] for line in output_lines
                if line and line.startswith("drwx")
            ]
            theme_predicted_outcome_counts = {}
            for folder in folder_names:
                parquet_path = f"{theme_predicted_outcome_counts_hdfs_path}/{folder}"
                theme_predicted_outcome_counts[
                    (folder.split('-')[0], folder.split('-')[1])] = self.session.read.parquet(parquet_path)
            loaded_refinery.theme_predicted_outcome_counts = theme_predicted_outcome_counts

            theme_counterfactual_outcome_counts_hdfs_path = f"suggestions_measurement__output/{self.run_id}/{self.run_id}/data/theme_counterfactual_outcome_counts"
            logger.debug('Loading theme counterfactual outcome counts data from: %s',
                         theme_counterfactual_outcome_counts_hdfs_path)
            output_lines = self._run_log_command(
                f'hdfs dfs -ls {theme_counterfactual_outcome_counts_hdfs_path}'
            )
            folder_names = [
                line.split()[-1].split('/')[-1] for line in output_lines
                if line and line.startswith("drwx")
            ]
            theme_counterfactual_outcome_counts = {}
            for folder in folder_names:
                parquet_path = f"{theme_counterfactual_outcome_counts_hdfs_path}/{folder}"
                theme_counterfactual_outcome_counts[
                    (folder.split('-')[0], folder.split('-')[1])] = self.session.read.parquet(parquet_path)
            loaded_refinery.theme_counterfactual_outcome_counts = theme_counterfactual_outcome_counts

        return loaded_refinery
