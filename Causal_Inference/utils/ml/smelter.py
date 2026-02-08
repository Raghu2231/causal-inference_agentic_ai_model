'''
TODO:
    - Add docstrings
    - Build unit tests
'''

from abc import ABC, abstractmethod
from functools import reduce
from typing import Callable, Sequence

from pyspark.sql import Column
import pyspark.sql.functions as F

from mlops_suggestion_measurement.suggestions_measurement import config, logger
from mlops_suggestion_measurement.suggestions_measurement.utils.spark import (
    PercentOtherHCPSSuggestedWindow,
    WindowSpecManager,
    GlobalWeeklyActivityWindow,
    RepWeeklyActivityWindow,
    HCPWeeklyActivityWindow
)

class Smelter(ABC):
    
    def __init__(
        self,
    ):
        logger.debug(f'Initialized {self.NAME} Smelter.')

    @property
    @abstractmethod
    def NAME(self) -> str:
        pass

    @abstractmethod
    def transform(self, df):
        pass


class PercentOtherHCPSSuggestedSmelter(Smelter):
    NAME: str = 'PercentOtherHCPSSuggestedSmelter'
    WINDOW_SPEC: WindowSpecManager = PercentOtherHCPSSuggestedWindow()

    @property
    def suggestions_column(self) -> Column:
        return (F.coalesce('suggestion_count', F.lit(0)) > 0).astype('double')

    def transform(self, df):
        total_rep_suggestions = F.sum(self.suggestions_column).over(self.WINDOW_SPEC.run())
        this_hcp_rep_suggestions = self.suggestions_column
        total_hcps = F.count(self.suggestions_column).over(self.WINDOW_SPEC.run())

        return (
            df
            .select([
                "*",
                F.coalesce(
                    (total_rep_suggestions - this_hcp_rep_suggestions) / (total_hcps - F.lit(1)),
                    F.lit(0.)
                ).alias("percent_other_hcps_suggested")
            ])
        )


class GlobalActivityIndexSmelter(Smelter):
    NAME: str = 'GlobalActivityIndexSmelter'
    WINDOW_SPEC: WindowSpecManager = GlobalWeeklyActivityWindow()

    def transform(
        self,
        sdf: config.Types.DataFrame,
        activity_to_index: str
    ) -> config.Types.DataFrame:
        logger.debug('Getting global activity index for observable: %s', activity_to_index)
        return (
            sdf
            .withColumn(
                f'{activity_to_index}_global_index',
                (
                    F
                    .sum(activity_to_index)
                    .over(
                        self.WINDOW_SPEC.run()
                    )
                    .cast('int')
                )
            )
        )
    

class RepActivityIndexSmelter(Smelter):
    NAME: str = 'RepActivityIndexSmelter'
    WINDOW_SPEC: WindowSpecManager = RepWeeklyActivityWindow()

    def transform(
        self,
        sdf: config.Types.DataFrame,
        activity_to_index: str
    ) -> config.Types.DataFrame:
        logger.debug('Getting global rep activity index for observable: %s', activity_to_index)
        return (
            sdf
            .withColumn(
                f'{activity_to_index}_rep_global_index',
                (
                    F
                    .sum(activity_to_index)
                    .over(
                        self.WINDOW_SPEC.run()
                    )
                    .cast('int')
                )
            )
        )
    
class HCPActivityIndexSmelter(Smelter):
    NAME: str = 'HCPActivityIndexSmelter'
    WINDOW_SPEC: WindowSpecManager = HCPWeeklyActivityWindow()

    def transform(
        self,
        sdf: config.Types.DataFrame,
        activity_to_index: str
    ) -> config.Types.DataFrame:
        logger.debug('Getting global HCP activity index for observable: %s', activity_to_index)
        return (
            sdf
            .withColumn(
                f'{activity_to_index}_hcp_global_index',
                (
                    F
                    .sum(activity_to_index)
                    .over(
                        self.WINDOW_SPEC.run()
                    )
                    .cast('int')
                )
            )
        )