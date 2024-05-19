
import pickle
import os
import numpy as np
import pandas as pd
from src.utils import load_object


try:
    model_path=os.path.join("artifacts","model.pkl")
    preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
    
    model=load_object(file_path=model_path)
    preprocessor=load_object(file_path=preprocessor_path)

except FileNotFoundError:
    print("Error: Model file 'model.pkl' not found.")
    

def main():

    input_data = pd.DataFrame({
        'Income': [20000],
        'Amount_Spent': [1000],
        'Children': [1],
        'Customer_Age': [40],
        'Total_Purchases': [1000],
        'TotalAcceptedCmp': [100],
        'Education': ['Graduate'],
        'Living_With': ['Alone'],
        'Family_Size': [3]
        })

        # Perform prediction using the loaded model
    data = preprocessor.transform(input_data)
    prediction = model.predict(data)[0]
        
    # Display prediction result
    
    if prediction == 1:
        print("Customers comes under Cluster 1, have the following attributes:")
        print("- Higher income")
        print("- Higher amount spent")
        print("- Single or parent of less than 3 kids")
        print("- Higher amount of purchases")
            
    elif prediction == 0:
        print("Customer comes under Cluster 0, have the following attributes:")
        print("- Lower income")
        print("- Lower amount spent")
        print("- Married and parent of more than 3 kids")
        print("- Lower amount of purchases")

if __name__ == '__main__':
    main()
