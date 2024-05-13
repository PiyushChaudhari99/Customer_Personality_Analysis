import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



@dataclass
class DataTransformationConfig:
     preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            numerical_columns = ['Year_Birth', 'Income', 'Kidhome',
       'Teenhome', 'Recency', 'MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
            

            categorical_columns = ['Education', 'Marital_Status', 
       'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2', 'Complain', 'Response']


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

            logging.info("Obtaining preprocessing object")

            #Reducing the nmber of columns by merging different columns
            train_df['Amount_Spent'] = train_df['MntWines'] + train_df['MntFruits'] + train_df['MntMeatProducts'] + train_df['MntFishProducts'] + train_df['MntSweetProducts'] + train_df['MntGoldProds']
            train_df['Living_With'] = train_df['Marital_Status'].replace({'Married':'Partner', 'Together':'Partner', 'Absurd':'Alone', 'Widow':'Alone', 'YOLO':'Alone', 'Divorced':'Alone', 'Single':'Alone'})
            train_df['Children'] = train_df['Kidhome'] + train_df['Teenhome']
            train_df['Family_Size'] = train_df['Living_With'].replace({'Alone': 1, 'Partner':2}) + train_df['Children']
            train_df['Education'] = train_df['Education'].replace({'Basic':'Undergraduate', '2n Cycle':'Undergraduate', 'Graduation':'Graduate', 'Master':'Postgraduate', 'PhD':'Postgraduate'})
            train_df['Customer_Age'] = (pd.Timestamp('now').year) - train_df['Year_Birth']
            train_df['Total_Purchases'] = train_df['NumWebPurchases'] + train_df['NumCatalogPurchases'] + train_df['NumStorePurchases'] + train_df['NumDealsPurchases']
            train_df['TotalAcceptedCmp'] = train_df['AcceptedCmp1'] + train_df['AcceptedCmp2'] + train_df['AcceptedCmp3'] + train_df['AcceptedCmp4'] + train_df['AcceptedCmp5']
            train_df=train_df.drop(columns=["Year_Birth","Marital_Status","ID","AcceptedCmp1" , "AcceptedCmp2", "AcceptedCmp3" , "AcceptedCmp4","AcceptedCmp5","NumWebVisitsMonth", "NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumDealsPurchases" , "Kidhome", "Teenhome","MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds",'Dt_Customer','Recency','Complain','Response'],axis=1)
            
            print(train_df)
            
            preprocessing_obj=self.get_data_transformer_object()


            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(train_df)
            input_feature_test_arr=preprocessing_obj.transform(test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(test_df)]

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
