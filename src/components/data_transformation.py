import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            numerical_columns = [ 'Income', 'Amount_Spent', 'Children', 'Customer_Age','Total_Purchases','TotalAcceptedCmp']
            

            categorical_columns = ['Education', 'Living_With','Family_Size']


            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]

            )
            return preprocessor
        
        except Exception as e:
             raise CustomException(sys,e)



    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)


            logging.info("Read train and test data completed")

            logging.info("Performing feature engineering on train data")

            # Feature Engineering on train data
            train_df['Amount_Spent'] = train_df['MntWines'] + train_df['MntFruits'] + train_df['MntMeatProducts'] + train_df['MntFishProducts'] + train_df['MntSweetProducts'] + train_df['MntGoldProds']
            train_df['Living_With'] = train_df['Marital_Status'].replace({'Married': 'Partner', 'Together': 'Partner', 'Absurd': 'Alone', 'Widow': 'Alone', 'YOLO': 'Alone', 'Divorced': 'Alone', 'Single': 'Alone'})
            train_df['Children'] = train_df['Kidhome'] + train_df['Teenhome']
            train_df['Family_Size'] = train_df['Living_With'].replace({'Alone': 1, 'Partner': 2}) + train_df['Children']
            train_df['Education'] = train_df['Education'].replace({'Basic': 'Undergraduate', '2n Cycle': 'Undergraduate', 'Graduation': 'Graduate', 'Master': 'Postgraduate', 'PhD': 'Postgraduate'})
            train_df['Customer_Age'] = (pd.Timestamp('now').year) - train_df['Year_Birth']
            train_df['Total_Purchases'] = train_df['NumWebPurchases'] + train_df['NumCatalogPurchases'] + train_df['NumStorePurchases'] + train_df['NumDealsPurchases']
            train_df['TotalAcceptedCmp'] = train_df['AcceptedCmp1'] + train_df['AcceptedCmp2'] + train_df['AcceptedCmp3'] + train_df['AcceptedCmp4'] + train_df['AcceptedCmp5']

            train_df_new = train_df[['Income', 'Amount_Spent', 'Children', 'Customer_Age', 'Total_Purchases', 'TotalAcceptedCmp', 'Education', 'Living_With', 'Family_Size']]

            logging.info("Performing feature engineering on test data")

            # Feature Engineering on test data
            test_df['Amount_Spent'] = test_df['MntWines'] + test_df['MntFruits'] + test_df['MntMeatProducts'] + test_df['MntFishProducts'] + test_df['MntSweetProducts'] + test_df['MntGoldProds']
            test_df['Living_With'] = test_df['Marital_Status'].replace({'Married': 'Partner', 'Together': 'Partner', 'Absurd': 'Alone', 'Widow': 'Alone', 'YOLO': 'Alone', 'Divorced': 'Alone', 'Single': 'Alone'})
            test_df['Children'] = test_df['Kidhome'] + test_df['Teenhome']
            test_df['Family_Size'] = test_df['Living_With'].replace({'Alone': 1, 'Partner': 2}) + test_df['Children']
            test_df['Education'] = test_df['Education'].replace({'Basic': 'Undergraduate', '2n Cycle': 'Undergraduate', 'Graduation': 'Graduate', 'Master': 'Postgraduate', 'PhD': 'Postgraduate'})
            test_df['Customer_Age'] = (pd.Timestamp('now').year) - test_df['Year_Birth']
            test_df['Total_Purchases'] = test_df['NumWebPurchases'] + test_df['NumCatalogPurchases'] + test_df['NumStorePurchases'] + test_df['NumDealsPurchases']
            test_df['TotalAcceptedCmp'] = test_df['AcceptedCmp1'] + test_df['AcceptedCmp2'] + test_df['AcceptedCmp3'] + test_df['AcceptedCmp4'] + test_df['AcceptedCmp5']

            test_df_new = test_df[['Income', 'Amount_Spent', 'Children', 'Customer_Age', 'Total_Purchases', 'TotalAcceptedCmp', 'Education', 'Living_With', 'Family_Size']]

            print(train_df_new)

            logging.info(f"Obtaining preprocessor object.")

            preprocessing_obj = self.get_data_transformer_object()

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(train_df_new)
            input_feature_test_arr = preprocessing_obj.transform(test_df_new)

            logging.info(f"preprocessing of the input feature done.")

            train_arr = np.c_[input_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr]

            print(pd.DataFrame(train_arr))

            logging.info(f"Saved preprocessing object.")


            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
