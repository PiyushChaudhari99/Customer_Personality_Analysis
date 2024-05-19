import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score



from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,X_test=(
                train_array,
                test_array
            )
            model = KMeans(n_clusters=2, random_state=42,n_init="auto")
            
            model_report:dict=evaluate_model(X_train=X_train,X_test=X_test,model=model)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            predicted=model.predict(X_test)

            model_silhouette_score = silhouette_score(X_test,predicted)
            return model_silhouette_score
  
        except Exception as e:
            raise CustomException(e,sys)