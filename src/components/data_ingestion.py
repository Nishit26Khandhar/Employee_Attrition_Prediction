import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
from src.components.Featur_engineering import FeatureEngineering

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    raw_data_source: str = os.path.join('notebook', 'data', 'Palo Alto Networks.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            df=pd.read_csv(self.ingestion_config.raw_data_source)
            logging.info('Read the dataset as dataframe')

              # 🔥 CLEAN TARGET COLUMN (MOST IMPORTANT)
            df['Attrition'] = df['Attrition'].astype(str).str.strip().str.lower()
            attrition_map = {'yes': 1, 'no': 0, 'y': 1, 'n': 0, '1': 1, '0': 0}
            df['Attrition'] = df['Attrition'].map(attrition_map)
            df = df.dropna(subset=['Attrition'])
            df['Attrition'] = df['Attrition'].astype(int)

            if df.shape[0] == 0:
                raise Exception("Dataset is empty after cleaning Attrition column")
 
            logging.info(f"Target distribution:\n{df['Attrition'].value_counts()}")

            df['OverTime'] = df['OverTime'].astype(str).str.strip().str.lower()
            df['OverTime'] = df['OverTime'].map({'yes': 1, 'no': 0, '1': 1, '0': 0})

            df = df.dropna(subset=['OverTime'])          # guard against bad values
            df['OverTime'] = df['OverTime'].astype(int)
        
            
            fe = FeatureEngineering()
            df = fe.engineer_features(df)
            logging.info("Feature engineering applied successful")

            # Split

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            train_set,test_set=train_test_split(df,test_size=0.2,stratify=df['Attrition'],random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)


            logging.info(f"Train shape: {train_set.shape} | Test shape: {test_set.shape}")
            logging.info("Data ingestion completed successfully")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)


    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

            



