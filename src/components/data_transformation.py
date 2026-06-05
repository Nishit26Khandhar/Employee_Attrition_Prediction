import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
    
    NUMERICAL_COLUMNS = [ 'Age','DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
                                 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
                                'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                                'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                                'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                                'YearsSinceLastPromotion', 'YearsWithCurrManager',
                                 # Engineered features
                                'IncomePerYearExp', 'PromotionDelay', 'EngagementScore',
                                'WorkloadStress', 'HighPerf_NoPromo','JobHopperIndex', 'LoyaltyScore', 'DissatisfactionRisk',
                                # OverTime is now 0/1 integer from ingestion, treat as numeric
                                'OverTime']
    CATEGORICAL_COLUMNS = ['BusinessTravel', 'Department', 'EducationField',
                           'Gender', 'JobRole', 'MaritalStatus'
                           ]
class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''This function si responsible for data trnasformation'''
        try:

            num_cols = self.data_transformation_config.NUMERICAL_COLUMNS
            cat_cols = self.data_transformation_config.CATEGORICAL_COLUMNS
        
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Numerical columns ({len(num_cols)}): {num_cols}")
            logging.info(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_cols),
                ("cat_pipelines",cat_pipeline,cat_cols)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            num_cols = self.data_transformation_config.NUMERICAL_COLUMNS
            cat_cols = self.data_transformation_config.CATEGORICAL_COLUMNS

            all_expected = num_cols + cat_cols + ['Attrition']
            missing_train = [c for c in all_expected if c not in train_df.columns]
            missing_test  = [c for c in all_expected if c not in test_df.columns]
            if missing_train:
                raise ValueError(f"Missing columns in train set: {missing_train}")
            if missing_test:
                raise ValueError(f"Missing columns in test set: {missing_test}")
 
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name="Attrition"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Before SMOTE → Class distribution: {dict(zip(*np.unique(target_feature_train_df, return_counts=True)))}")

            smote_tomek = SMOTETomek(
                smote=SMOTE(sampling_strategy=0.9, random_state=42),
                random_state=42
            )
            input_feature_train_arr, target_feature_train_resampled = smote_tomek.fit_resample(
                input_feature_train_arr,
                target_feature_train_df
            )

            logging.info(f"After SMOTETomek  → Class distribution: {dict(zip(*np.unique(target_feature_train_resampled, return_counts=True)))}")

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_resampled)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

           
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            logging.info("Preprocessor saved to artifacts/preprocessor.pkl")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
