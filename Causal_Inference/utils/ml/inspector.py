from abc import ABC, abstractmethod
from itertools import chain
from typing import Callable, Optional, Sequence
from functools import reduce
from uuid import uuid4
from pathlib import Path

# from plotnine import (
#     ggplot,
#     aes,
#     geom_bar,
#     geom_hline,
#     labs,
#     theme,
#     coord_flip,
#     ggtitle
# )
import pandas as pd
from pyspark.sql import functions as F
from pyspark.ml.regression import LinearRegressionModel, GBTRegressionModel
from pyspark.ml.feature import VectorAssembler, StringIndexerModel, StandardScalerModel, OneHotEncoderModel
from pyspark.ml.pipeline import PipelineModel
from .refinery import Refinery
from ... import config, logger


class Inspector(ABC):

    def __init__(
            self,
            refinery: Refinery
    ) -> None:
        self.id: str = str(uuid4())
        self.refinery: Refinery = refinery

        logger.info(
            'Initializing %s inspector for refinery with ID: %s',
            self.NAME,
            self.id
        )

    @property
    @abstractmethod
    def NAME(self) -> str:
        pass

    @abstractmethod
    def get_pipeline(
            self,
    ) -> Sequence[Callable]:
        pass

    def save(self) -> config.Types.Self:
        model_config = self.refinery.alchemists['PathA_PropensityModel'].model_config

        output_directory = Path(model_config.path.viz)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / f'{self.NAME}_{self.id}.png'

        logger.debug('Saving %s inspector: %s', self.NAME, self.id)
        self.plot.save(
            filename=output_path,
            format='png'
        )
        logger.info('Plot successfully saved to %s', output_path)
        return self

    def run(
            self
    ) -> config.Types.Self:
        logger.debug('Running %s inspector: %s', self.NAME, self.id)

        pipeline = self.get_pipeline()
        if hasattr(self, 'plot'):
            pipeline += [self.save]

        return reduce(
            lambda acc, func: func(),
            pipeline,
            self
        )


# Summary Data

class SummaryData(Inspector):

    def __init__(
            self,
            refinery: Refinery,
            report_dimension: list = None,
            join_dimension: list = [
                'year',
                'month'
            ]
    ) -> None:
        super().__init__(refinery)
        self.report_dimension: list = report_dimension
        self.join_dimension: list = join_dimension
        self.summary_data: Optional[pd.DataFrame] = None

    @property
    def NAME(self) -> str:
        return 'summary_data'

    def _get_summary_data(self) -> config.Types.Self:
        """
        Creates the pandas df summary_data.
        """
        if self.report_dimension == None:
            self.report_dimension = [
                F.year('week_end_date').alias('year'),
                F.month('week_end_date').alias('month'),
                # 'week_end_date',
                # 'abbott_customer_id'
                # 'rep_id',
            ]

        self.summary_data: pd.DataFrame = (
            self.refinery
            .alchemists
            ['PathA_TreatmentEffectModel']
            .feature_engineer
            .df
            # TODO: Add args for suggestion, outcome, action, type
            .filter(
                F.col('suggestion_type') == 'call'
            )
            .filter(
                F.col('outcome_type') == 'nbrx'
            )
            .filter(
                F.col('action_type') == 'call'
            )
            .groupBy(*[
                *self.report_dimension,
                # 'suggestion_type',
                # 'outcome_type',
                # 'action_type'
            ])
            .agg(*[
                F.sum('outcome_count').alias('observed_outcome'),
                F.sum('action_count').alias('observed_action'),
                # F.sum('suggestion_count').alias('observed_suggestion')
            ])
            .join(
                (
                    self.refinery
                    .incremental_outcome
                    .groupBy(*[
                        *self.report_dimension,
                    ])
                    .agg(*[
                        F.sum('incremental_outcome').alias('incremental_outcome'),
                        F.sum('predicted_outcome_count').alias('predicted_outcome')
                    ])
                ),
                how='inner',
                on=[
                    # *self.report_dimension,
                    *self.join_dimension,
                ]
            )
            .join(
                (
                    self.refinery
                    .incremental_action
                    .groupBy(*[
                        *self.report_dimension,
                    ])
                    .agg(*[
                        F.sum('incremental_action').alias('incremental_action'),
                        F.sum('predicted_action_count').alias('predicted_action')
                    ])
                ),
                how='inner',
                on=[
                    # *self.report_dimension,
                    *self.join_dimension,
                ]
            )
            .toPandas()
        )

    def get_pipeline(self) -> Sequence[Callable]:
        return [self._get_summary_data]


class ActualVsPredicted(SummaryData):

    @property
    def NAME(self) -> str:
        return 'actual_vs_predicted'

    @property
    @abstractmethod
    def X_COLUMN(self) -> str:
        """
        Column to be used for the x-axis in the plot.
        """
        pass

    @property
    @abstractmethod
    def Y_COLUMN(self) -> str:
        """
        Column to be used for the y-axis in the plot.
        """
        pass

    def _get_plot(self) -> config.Types.Self:
        """
        Creates the actual vs predicted plot using the specified columns for the x-axis and y-axis.
        """
        logger.info(f"Generating plot for {self.X_COLUMN} vs {self.Y_COLUMN}")

        self.plot = (
                ggplot(self.summary_data)
                + aes(
            x=self.X_COLUMN,
            y=self.Y_COLUMN
        )
                + geom_hline(
            yintercept=0
        )
                + geom_vline(
            xintercept=0
        )
                + geom_abline(
            aes(
                slope=1,
                intercept=0
            ),
            linetype='dashed'
        )
                + geom_point()
                + geom_smooth(
            method='ols'
        )
                + theme(
            figure_size=(8, 6)
        )
                + labs(
            x=self.X_COLUMN.replace('_', ' ').title(),
            y=self.Y_COLUMN.replace('_', ' ').title()
        )
                + ggtitle(f'Predicted vs Observed Outcomes\nDimension: {self.join_dimension}')
        )

        logger.info("Plot generation complete.")
        return self

    def get_pipeline(self) -> Sequence[Callable]:
        return super().get_pipeline() + [self._get_plot]


class ActualVsPredictedAction(ActualVsPredicted):

    @property
    def NAME(self) -> str:
        return 'actual_vs_predicted_action'

    @property
    def X_COLUMN(self) -> str:
        return 'predicted_action'

    @property
    def Y_COLUMN(self) -> str:
        return 'observed_action'


class ActualVsPredictedOutcome(ActualVsPredicted):

    @property
    def NAME(self) -> str:
        return 'actual_vs_predicted_outcome'

    @property
    def X_COLUMN(self) -> str:
        return 'predicted_outcome'

    @property
    def Y_COLUMN(self) -> str:
        return 'observed_outcome'


class ModelSummary(SummaryData):

    @property
    def NAME(self) -> str:
        return "model_summary"

    def _get_plot(self) -> config.Types.Self:
        """
        Creates the model summary plot.
        """
        logger.info("Generating plot")

        self.plot = (
            self.summary_data
            .melt(
                id_vars=[
                    # *self.report_dimension,
                    *self.join_dimension,
                    # 'suggestion_type',
                    # 'action_type',
                    # 'outcome_type'
                ]
            )
            .assign(
                year_month=lambda df: (
                        df['year'].astype('str') + '_' + df['month'].astype('str').str.zfill(2)
                )
            )
            .pipe(lambda df: (
                    ggplot(df)
                    + aes(
                x='year_month',
                y='value'
            )
                    + facet_wrap(
                '~variable',
                ncol=2,
                scales='free_x',
                labeller=lambda _x: f'Observable: {_x}'
            )
                    + geom_bar(
                stat='identity',
                fill=sm.config.Colors.onesix_purple
            )
                    + geom_hline(
                yintercept=0
            )
                    + theme(
                figure_size=(16, 16),
                panel_spacing=0.5
            )
                    + coord_flip()
                    + ggtitle('Monthly Overview: Incremental, Observed, and Predicted Quantities')
            ))
        )

        logger.info("Plot generation complete.")
        return self

    def get_pipeline(self) -> Sequence[Callable]:
        return super().get_pipeline() + [self._get_plot]


# MODEL FEATURE CONTRIBUTION

## Base Class
class ModelFeatureContribution(Inspector):
    NAME: str = "Model Feature Contribution"

    def __init__(
            self,
            refinery: Refinery
    ) -> None:
        self.pipeline: Optional[Sequence[object]] = None
        self.feature_contributions: Optional[pd.DataFrame] = None
        super().__init__(refinery)

    @staticmethod
    def __get_one_hot_encoded_features(
            pipeline: PipelineModel
    ) -> Sequence[str]:
        logger.debug("Getting one-hot encoded features from pipeline: %s", pipeline)
        ohe_features: Sequence[str] = list(chain.from_iterable([
            [
                "_".join([
                    stage.getInputCol(),
                    _value
                ])
                for _value in stage.labels[:-1]
            ]
            for stage in pipeline
        ]))
        logger.debug("Found %d one-hot encoded features: %s", len(ohe_features), ohe_features)
        return ohe_features

    def _get_flat_pipeline(
            self
    ) -> config.Types.Self:
        logger.debug("Flattening model pipeline for model: %s", self.MODEL_NAME)
        assembler: VectorAssembler = (
            self.refinery["assembler"]
            # .alchemists
            # [self.MODEL_NAME]
            # .feature_engineer
            # .assembler
        )
        if hasattr(assembler, "stages"):
            logger.debug("One-hot encoding detected.")
            assembler_stages: Sequence[object] = (
                assembler
                .stages
            )
            logger.debug("Assembler stages: %s", assembler_stages)

            self.pipeline: Sequence[object] = [
                *[
                    __stage
                    for _stage in assembler_stages
                    if not any(isinstance(_stage, _type) for _type in [
                        VectorAssembler,
                        StandardScalerModel,
                        OneHotEncoderModel
                    ])
                    for __stage in _stage.stages
                    if isinstance(__stage, StringIndexerModel)
                ],
                *[
                    _stage for _stage in assembler_stages
                    if isinstance(_stage, VectorAssembler)
                ],
                (
                    self.refinery["trained_model"]
                    # .alchemists
                    # [self.MODEL_NAME]
                    # .trained_model
                    .stages
                    [0]
                )
            ]

            logger.debug("Flat pipeline has %d stages.", len(self.pipeline))

            return self

        self.pipeline: Sequence[object] = [
            assembler,
            (
                self.refinery["trained_model"]
                # .alchemists
                # [self.MODEL_NAME]
                # .trained_model
                .stages
                [0]
            )
        ]

        logger.debug("Flat pipeline has %d stages.", len(self.pipeline))

        return self

    def _get_feature_contributions(
            self
    ) -> config.Types.Self:
        model = self.pipeline[-1]
        vector_assembler: VectorAssembler = (
            self.pipeline[-2]
            if not isinstance(self.pipeline[-2], StandardScalerModel) else
            self.pipeline[-3]
        )

        try:
            feature_contributions: Sequence[float] = config.FEATURE_CONTRIBUTION_EXTRACTORS[type(model).__name__](model)
        except KeyError:
            raise ValueError(f"Unsupported model type: {type(model).__name__}")

        vector_assembler_features: Sequence[str] = vector_assembler.getInputCols()

        logger.debug("Looking for one-hot encoded features in pipeline: %s", self.pipeline)

        ohe_features = (
            self.__get_one_hot_encoded_features(
                pipeline=(
                    # Exclude vector assembler and trained model
                    filter(
                        lambda _stage: isinstance(_stage, StringIndexerModel),
                        self.pipeline,
                    )
                )
            )
            if any([
                isinstance(_stage, StringIndexerModel)
                for _stage in self.pipeline
            ]) else
            []
        )

        logger.debug("Found %d one-hot encoded features: %s", len(ohe_features), ohe_features)

        all_feature_labels = [
            *ohe_features,
            *[
                _feature
                for _feature in vector_assembler_features
                if not _feature.endswith("vector")
            ]
        ]
        logger.debug("Found %d total features: %s", len(all_feature_labels), all_feature_labels)

        assert len(feature_contributions) == len(all_feature_labels), (
            f"Expected the same number of importances and labels, but found {len(feature_importances)} "
            f"and {len(all_feature_labels)}, respectively."
        )

        logger.debug("Getting feature importances dataframe.")

        self.feature_contributions: pd.DataFrame = (
            pd.DataFrame({
                "feature": all_feature_labels,
                "contribution": feature_contributions,
            })
            .sort_values(by="contribution", ascending=True)
        )

        return self

    def _plot_feature_contributions(
            self
    ) -> config.Types.Self:
        self.feature_contributions["feature"] = pd.Categorical(
            self.feature_contributions["feature"],
            categories=self.feature_contributions["feature"],
            ordered=True,
        )
        logger.debug(
            "Plotting %d feature importances for model: %s",
            len(self.feature_contributions),
            self.MODEL_NAME,
        )

        self.plot = (
                self.feature_contributions
                .pipe(lambda df: (
                        ggplot(df)
                        + aes(
                    x="feature",
                    y="contribution",
                )
                        + geom_bar(
                    stat="identity",
                    fill=config.Colors.onesix_purple,
                )
                        + geom_hline(
                    aes(
                        yintercept=0
                    )
                )
                        + coord_flip()
                        + theme(
                    figure_size=(
                        9, 16
                    )
                )
                ))
                + labs(
            x="Feature",
            y="Contribution",
        )
                + ggtitle(f"Feature Contribution: {self.MODEL_NAME}")
        )

        return self

    def model_matrix(
            self,
            filters: Sequence[object] = tuple(),
            order_by: Sequence[object] = tuple(),
            include_index: bool = False,
    ) -> config.Types.DataFrame:
        """ Gets the model matrix used for model training

        Optional filters argument allows you to provide something like

        ```
        .model_matrix(filters=[
            F.col("week_end_date") == "2023-01-01",
            F.col("rep_id") == ...
        ])
        ```

        Filters are AND'd together

        TODO: Tear this out, move to the feature engineers or alchemist. The coefficients from ModelCoefficients are why
        it's currently implemented here.
        """
        if not getattr(self, "pipeline"):
            self._get_flat_pipeline()
        if not getattr(self, "model_contributions"):
            self._get_model_contributions()

        if include_index and "PathA" in self.MODEL_NAME:
            index_cols = ['abbott_customer_id', 'rep_id', 'week_end_date']
        elif include_index and "PathB" in self.MODEL_NAME:
            index_cols = ['abbott_customer_id', 'week_end_date']
        elif include_index:
            raise ValueError(f"Unknown MODEL_NAME: {self.MODEL_NAME}, cannot retrieve model matrix index")
        else:
            index_cols = []

        feature_engineer_df_ = (
            self.refinery
            .alchemists[self.MODEL_NAME]
            .feature_engineer
            .df_  # FIXME Do not use this attribute directly.
        )

        filtered_df = reduce(
            lambda df, filter: df.where(filter),
            filters,
            feature_engineer_df_,
        )

        ordered_df = filtered_df.orderBy(*order_by) if order_by else filtered_df

        return (
            ordered_df
            .rdd
            .map(
                lambda x: [str(x[key]) for key in index_cols] + [float(y) for y in x['features']]
            )
            .toDF(
                index_cols +
                list(self.model_coefficients["feature"])
            )
        )

    def get_pipeline(
            self
    ) -> Sequence[Callable]:
        return [
            self._get_flat_pipeline,
            self._get_feature_contributions,
            #s elf._plot_feature_contributions,
        ]


class PathAPropensityModelFeatureContribution(ModelFeatureContribution):
    MODEL_NAME: str = 'PathA_PropensityModel'


class PathBPropensityModelFeatureContribution(ModelFeatureContribution):
    MODEL_NAME: str = 'PathB_PropensityModel'


class PathATreatmentEffectModelFeatureContribution(ModelFeatureContribution):
    MODEL_NAME: str = 'PathA_TreatmentEffectModel'


class PathBTreatmentEffectModelFeatureContributionNBRx(ModelFeatureContribution):
    MODEL_NAME: str = 'PathB_TreatmentEffectModel__nbrx'


class PathBTreatmentEffectModelFeatureContributionTRx(ModelFeatureContribution):
    MODEL_NAME: str = 'PathB_TreatmentEffectModel__trx'

