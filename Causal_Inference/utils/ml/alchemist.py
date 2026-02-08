'''
This module provides the methodology for all of the 4 of the basic machine learning models:
- Path A
  - Propensity Model
  - Treatment Effect Model
- Path B
  - Propensity Model
  - Treatment Effect Model

The key method can be seen in `Alchemist.run()`:

TODO:
    - Build unit tests
    - Add docstring
'''
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Union
from uuid import uuid4
import json
from pathlib import Path
import subprocess
from pathlib import PurePath
import pandas as pd
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from ... import config, logger
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator

# from .engineer import FeatureEngineer


class Alchemist(ABC):
    FEATURES_COLUMN: str = config.Column.features_column
    MODEL_PARAMS: Optional[Dict] = None

    def __init__(
            self,
            # feature_engineer: FeatureEngineer,
            model_config: config.Types.ModelConfig,
            df,
            pred_df
    ):
        # TODO: Abstract validation logic into seprate class
        # assert isinstance(feature_engineer, FeatureEngineer), (
        #     'Expected feature_engineer to be of type FeatureEngineer, '
        #     f'but got {type(feature_engineer)}'
        # )
        assert self.MODEL_TYPE in config.ALLOWED_MODEL_TYPES, (
            f'Expected model type to be one of {config.ALLOWED_MODEL_TYPES}, '
            f'but got {type(self.MODEL_TYPE)}'
        )
        assert model_config.model_type == self.MODEL_TYPE, (
            f'Expected model_config to be of type {self.MODEL_TYPE}, '
            f'but got {model_config.model_type}'
        )
        # assert feature_engineer.MODEL_TYPE == model_config.model_type, (
        #     f'Expected feature_engineer to be of type {model_config.model_type}, '
        #     f'but got {feature_engineer.model_type}'
        # )
        assert self.MODEL_OBJECT in config.ALLOWED_MODEL_OBJECTS.values(), (
            f'Expected model object to be one of {config.ALLOWED_MODEL_OBJECTS.values()}, '
            f'but got {self.MODEL_OBJECT}'
        )

        self.feature_engineer = df
        self.prediction_df = pred_df
        self.model_config: config.Types.ModelConfig = model_config
        self.model_id: str = self.model_config.model_id
        self.feature_importance = None
        self.model_metric = None
        logger.info(f'Initializing {self.MODEL_TYPE} model with ID: {self.model_id}')

    @property
    @abstractmethod
    def LABEL_COL(self):
        pass

    @property
    @abstractmethod
    def MODEL_OBJECT(self):
        pass

    @property
    @abstractmethod
    def MODEL_TYPE(self):
        pass

    def _make_model_pipeline(
            self,
    ) -> config.Types.Self:
        model_params: Dict = self.MODEL_PARAMS or {}
        logger.debug('Getting model pipeline with model params: %s', model_params)
        logger.debug('Using label column: %s', self.LABEL_COL)

        self.pipeline: Pipeline = Pipeline(
            stages=[
                self.MODEL_OBJECT(
                    featuresCol=self.FEATURES_COLUMN,
                    labelCol=self.LABEL_COL,
                    seed=7117,
                    **model_params,
                )
            ]
        )

        return self

    def _train_model(
            self,
    ) -> config.Types.Self:
        if not hasattr(self, 'pipeline'):
            raise AttributeError('Model pipeline not found. Please run _get_model_pipeline() first.')

        logger.debug('Running feature engineer')
        _df = self.feature_engineer

        logger.debug('Training model: %s', self.model_id)
        self.trained_model: self.MODEL_OBJECT = (
            self.pipeline
            .fit(_df)
        )
        logger.debug('Finished training model: %s', self.model_id)
        return self

    def _post_training_evaluation(
            self,
    ) -> config.Types.Self:
        logger.debug('No post-training evaluation set for: %s', self.model_id)
        return self

    @property
    def model_write_path(self) -> str:
        return str(self.model_config.path.model_write_path(
            model_type=self.MODEL_TYPE,
            model_id=self.model_id,
        ))

    def _save_model(
            self,
    ) -> config.Types.Self:
        logger.debug('Saving model to: %s', self.model_write_path)
        (
            self.trained_model
            .write()
            .overwrite()
            .save(str(self.model_write_path))
        )

        return self

    def _make_predictions(
            self
    ) -> config.Types.Self:
        self.data_with_predictions: config.Types.DataFrame = (
            self.prediction_df
            .transform(
                self.trained_model.transform
            )
        )

        return self

    def _post_inference_processing(
            self,
    ) -> config.Types.Self:
        logger.debug('No post-inference evaluation set for: %s', self.model_id)
        return self

    @property
    def data_write_path(self) -> str:
        return str(self.model_config.path.data_write_path(
            model_type=self.MODEL_TYPE,
            model_id=self.model_id,
        ))

    def _save_data(
            self
    ):
        # TODO: Unit test
        assert hasattr(self, 'data_with_predictions'), (
            'No data to save. Please run the feature engineer first.'
        )
        if not 'propensity' in self.MODEL_TYPE.lower():
            logger.debug('Not propensity data so not saving model: %s', self.model_id)
            return self

        # logger.debug('Saving data to: %s', self.data_write_path)

        maybe_columns_to_save: Sequence[str] = [
            "rep_id",
            "abbott_customer_id",
            "week_end_date",
            self.model_config.propensity_column
        ]
        columns_to_save: Sequence[str] = list(
            set(maybe_columns_to_save)
            .intersection(
                set(self.data_with_predictions.columns)
            )
        )
        logger.debug('Columns to save: %s', columns_to_save)
        self.data_with_predictions = self.data_with_predictions.select(*columns_to_save)

        # (
        #     self.data_with_predictions
        #     .select(*columns_to_save)
        #     .write
        #     .mode('overwrite')
        #     .parquet(self.data_write_path)
        # )

        return self

    @property
    def model_config_write_path(self) -> str:
        if self.MODEL_TYPE == 'PathB_TreatmentEffectModel':
            model_type = self.MODEL_TYPE + '__' + self.model_config.outcome_type
            return str(self.model_config.path.model_config_write_path(
                model_type=model_type
            ))
        return str(self.model_config.path.model_config_write_path(
            model_type=self.MODEL_TYPE,
        ))

    def _save_model_config(
            self
    ) -> config.Types.Self:
        # TODO: Unit test
        assert hasattr(self, 'model_config'), (
            'No model_config to save.'
        )

        logger.debug('Saving model_config to: %s', self.model_config_write_path)

        # Create local temporary file
        local_dir = Path("temp/model_config")  # Temporary local directory
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / "model_config.json"

        with open(local_file, "w") as json_file:
            json.dump(self.model_config, json_file, default=lambda obj: obj.__dict__, indent=4)

        # Define HDFS path
        hdfs_path: PurePath = self.model_config_write_path

        # Use subprocess to copy to HDFS
        subprocess.run(["hdfs", "dfs", "-put", "-f", str(local_file), hdfs_path], check=True)

        logger.debug(f"Saved model_config to HDFS at: {hdfs_path}")

        # Cleanup: Remove the local file
        if local_file.exists():
            local_file.unlink()
            logger.debug(f"Deleted local temp file: {local_file}")

        # Cleanup: Remove the directory if empty
        if local_dir.exists() and not any(local_dir.iterdir()):
            local_dir.rmdir()
            logger.debug(f"Deleted empty temp directory: {local_dir}")

        return self
        
    def _model_metric(
            self,
    ):
        from pyspark.sql import Row

        try:
            from pyspark.sql.functions import round as spark_round

            self.data_with_predictions = self.data_with_predictions.withColumn(
                "rounded_prediction", spark_round("prediction")
            )
            
            multi_evaluator = MulticlassClassificationEvaluator(
                labelCol=self.LABEL_COL,
                predictionCol="rounded_prediction"
            )
            accuracy = multi_evaluator.evaluate(self.data_with_predictions, {multi_evaluator.metricName: "accuracy"})
            precision = multi_evaluator.evaluate(self.data_with_predictions, {multi_evaluator.metricName: "weightedPrecision"})
            recall = multi_evaluator.evaluate(self.data_with_predictions, {multi_evaluator.metricName: "weightedRecall"})
            f1 = multi_evaluator.evaluate(self.data_with_predictions, {multi_evaluator.metricName: "f1"})
        except Exception as e:
            logger.warning(f"Multiclass metrics failed: {e}")
            accuracy = precision = recall = f1 = None

        # === Save metrics to DataFrame ===
        #metrics_row = Row("accuracy", "precision", "recall", "f1")
        #self.model_metric = self.data_with_predictions.sparkSession.createDataFrame([
        #    metrics_row(accuracy, precision, recall, f1)
        #])
        self.model_metric = self.data_with_predictions.sparkSession.createDataFrame([
            {
                "accuracy": accuracy or 0.0,
                "precision": precision or 0.0,
                "recall": recall or 0.0,
                "f1": f1 or 0.0
            }
        ])

        # === Feature Importances ===
        try:
            model_stage = self.trained_model.stages[-1]
            if hasattr(model_stage, "featureImportances"):
                importances = model_stage.featureImportances.toArray()

                # Step 1: Get feature names from VectorAssembler
                assembler_stage = next(
                    (s for s in self.trained_model.stages if s.__class__.__name__ == "VectorAssembler"),
                    None
                )

                if assembler_stage is None:
                    raise RuntimeError("VectorAssembler stage not found in pipeline. Cannot extract feature names.")

                input_features = assembler_stage.getInputCols()

                # Step 2: Zip feature names with importances
                feature_data = list(zip(input_features, importances))
                spark = self.data_with_predictions.sparkSession

                # Step 3: Create Spark DataFrame
                self.feature_importance = spark.createDataFrame(
                    feature_data,
                    schema=["feature_name", "importance"]
                ).orderBy(F.col("importance").desc())

            else:
                logger.info("Model does not support feature importances.")
                self.feature_importance = None

            
        except Exception as e:
            logger.warning(f"Failed to extract feature importances: {e}")
            self.feature_importance = None

        return self

    def run(
            self,
    ):
        logger.info(f'Running {self.MODEL_TYPE} model...')
        (
            self
            # ._save_model_config()
            ._make_model_pipeline()
            ._train_model()
            ._post_training_evaluation()
            # ._save_model()
            ._make_predictions()
            ._post_inference_processing()
            ._save_data()
            # ._model_metric()
        )
        logger.info(f'{self.MODEL_TYPE} model training complete!')
        return self.data_with_predictions, self.trained_model

    def prediction(
            self,
    ) -> None:
        logger.info(f'Running {self.MODEL_TYPE} model...')
        (
            self
            ._make_predictions()
            ._post_inference_processing()
            ._save_data()
        )
        logger.info(f'{self.MODEL_TYPE} model prediction complete!')


# Propensity Models

class PropensityAlchemist(Alchemist):
    def _post_inference_processing(self) -> config.Types.Self:
        if isinstance(self.MODEL_OBJECT(), config.ALLOWED_MODEL_OBJECTS['LogisticRegression']):
            logger.debug('Adding predicted propensity column to data from logistic regression model.')
            self.data_with_predictions: config.Types.DataFrame = (
                self.data_with_predictions
                .withColumn(
                    self.model_config.propensity_column,
                    (
                        vector_to_array('probability')
                        [1]
                    )
                )
            )
            return self
        if isinstance(self.MODEL_OBJECT(), config.ALLOWED_MODEL_OBJECTS['LinearRegression']):
            logger.debug('Adding predicted propensity column to data from linear regression model.')
            self.data_with_predictions: config.Types.DataFrame = (
                self.data_with_predictions
                .withColumn(
                    self.model_config.propensity_column,
                    F.column('prediction')
                )
            )
            return self
        if isinstance(self.MODEL_OBJECT(), config.ALLOWED_MODEL_OBJECTS['GBTRegressor']):
            logger.debug('Adding predicted propensity column to data from GBTRegressor model.')
            self.data_with_predictions: config.Types.DataFrame = (
                self.data_with_predictions
                .withColumn(
                    self.model_config.propensity_column,
                    F.column('prediction')
                )
            )
            return self

        raise RuntimeError('Model object not recognized. Unable to add propensity column.')


class PathAPropensityAlchemist(PropensityAlchemist):
    LABEL_COL: str = 'suggestion_count'
    MODEL_OBJECT: object = (
        config
        .ALLOWED_MODEL_OBJECTS
            # ['LinearRegression']
        ['GBTRegressor']
    )
    MODEL_PARAMS: Dict[str, Union[float, str]] = {
        # 'regParam': 0.0,
        # 'solver': 'normal',
        # 'standardization': False
    }
    MODEL_TYPE: str = 'PathA_PropensityModel'


class PathBPropensityAlchemist(PropensityAlchemist):
    LABEL_COL: str = 'action_count'
    MODEL_OBJECT: object = (
        config
        .ALLOWED_MODEL_OBJECTS
            # ['LinearRegression']
        ['GBTRegressor']
    )
    MODEL_PARAMS: Dict[str, Union[float, str]] = {
        # 'regParam': 0.00,
        # 'solver': 'normal',
        # 'standardization': False
    }
    MODEL_TYPE: str = 'PathB_PropensityModel'


# Treatment Effect Models

class PathATreatmentEffectAlchemist(Alchemist):
    LABEL_COL: str = 'action_count'
    MODEL_OBJECT: object = (
        config
        .ALLOWED_MODEL_OBJECTS
        ['GBTRegressor']
    )
    MODEL_PARAMS: Dict[str, Union[float, str]] = {
        # 'regParam': 0.0,
        # 'solver': 'normal',
        # 'standardization': False
    }

    MODEL_TYPE: str = 'PathA_TreatmentEffectModel'


class PathBTreatmentEffectAlchemist(Alchemist):
    LABEL_COL: str = 'outcome_count'
    MODEL_OBJECT: object = (
        config
        .ALLOWED_MODEL_OBJECTS
            # ['LinearRegression']
        ['GBTRegressor']
    )
    MODEL_PARAMS: Dict[str, Union[float, str]] = {
        # 'regParam': 0.00,
        # 'solver': 'normal',
        # 'standardization': False
    }
    MODEL_TYPE: str = 'PathB_TreatmentEffectModel'