# src/components/Featur_engineering.py

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class FeatureEngineeringConfig:
    pass  # No file paths needed; operates in-memory on DataFrames


class FeatureEngineering:
    def __init__(self):
        self.feature_engineering_config = FeatureEngineeringConfig()

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds engineered features to the DataFrame.
        Call this AFTER loading raw data, BEFORE transformation.
        """
        try:
            logging.info("Starting feature engineering")

            required_cols = [
                'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
                'YearsSinceLastPromotion', 'JobSatisfaction', 'EnvironmentSatisfaction',
                'RelationshipSatisfaction', 'OverTime', 'WorkLifeBalance',
                'PerformanceRating'
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns for feature engineering: {missing}")
 


            # 1. Income-to-experience ratio
            df['IncomePerYearExp'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)

            # 2. Promotion delay
            df['PromotionDelay'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']

            # 3. Engagement composite (avg of 1–5 satisfaction scales)
            df['EngagementScore'] = (
                df['JobSatisfaction'] +
                df['EnvironmentSatisfaction'] +
                df['RelationshipSatisfaction']
            ) / 3

            # 4. Workload stress flag
            # OverTime must already be binary (1/0) by this point
            df['WorkloadStress'] = (
                (df['OverTime'] == 1) & (df['WorkLifeBalance'] <= 2)
            ).astype(int)

            # 5. High performer with no recent promotion
            df['HighPerf_NoPromo'] = np.where(
                (df['PerformanceRating'] >= 4) & (df['YearsSinceLastPromotion'] > 3),
                1, 0
            )
            df['JobHopperIndex'] = df['NumCompaniesWorked'] / (df['Age'] - 17 + 1)
 
            # 7. Loyalty score — long tenure at current company relative to total career
            #    Low loyalty score = more likely to leave
            df['LoyaltyScore'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
 
            # 8. Dissatisfaction risk — combines low job involvement + low stock options
            #    + high distance from home: all known attrition drivers
            df['DissatisfactionRisk'] = (
                (df['JobInvolvement'] <= 2).astype(int) +
                (df['StockOptionLevel'] == 0).astype(int) +
                (df['DistanceFromHome'] > 15).astype(int)
            )

            logging.info(f"Feature engineering complete. New shape: {df.shape}")
            logging.info(f"New columns added: IncomePerYearExp, PromotionDelay, "
                         f"EngagementScore, WorkloadStress, HighPerf_NoPromo , JobHopperIndex, LoyaltyScore, DissatisfactionRisk")

            return df

        except Exception as e:
            raise CustomException(e, sys)