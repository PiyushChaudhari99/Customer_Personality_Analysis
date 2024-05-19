import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
from src.utils import load_object

# Load the pre-trained machine learning model


try:
    model_path=os.path.join("artifacts","model.pkl")
    preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
    
    model=load_object(file_path=model_path)
    preprocessor=load_object(file_path=preprocessor_path)


except FileNotFoundError:
    st.error("Error: Model file 'model.pkl' not found.")
    st.stop()


st.set_page_config(
    page_title="Customer Segmentation Prediction",
    page_icon=":bar_chart:",
    layout="wide",  # Wide layout for better spacing
    initial_sidebar_state="expanded",  # Expanded sidebar by default
)
custom_css = """
<style>
    body {
        font-family: Arial, sans-serif;
        background-color: #ffffff;
    }
    .st-dcuhf.st-dcvKpV.st-dcuhf.st-cUjEP.st-ekZUXy.st-elOvoR.st-dcuhf.st-dcvKpV.st-dcuhf.st-dcvKpV {
        color: yellow !important;  
   }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Define the main function for the Streamlit app
def main():
    
    st.title('Customer Segmentation Prediction')
    st.header('Enter Customer Features')
    # Input form for user to enter features
    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox('Education', options=['Undergraduate', 'Graduate', 'Postgraduate'], key='education_selectbox')
    
    with col2:
        living_with = st.selectbox('Living With', options=['Alone', 'Partner'],key='living_with_selectbox')


    income = st.number_input('Income', value=0.0)
    amount_spent = st.number_input('Amount Spent', value=0.0)
    children = st.number_input('Children', value=0)
    family_size = st.number_input('Family Size', value=1)
    customer_age = st.number_input('Customer Age', value=0)
    total_purchases = st.number_input('Total Purchases', value=0)
    total_accepted_cmp = st.number_input('Total Accepted Cmp', value=0)


    # Predict button to trigger prediction
    if st.button('Predict'):
        # Construct a DataFrame with the input features
        input_data = pd.DataFrame({
            'Income': [income],
            'Amount_Spent': [amount_spent],
            'Children': [children],
            'Customer_Age': [customer_age],
            'Total_Purchases': [total_purchases],
            'TotalAcceptedCmp': [total_accepted_cmp],
            'Education': [education],
            'Living_With': [living_with],
            'Family_Size': [family_size]
        })

        # Perform prediction using the loaded model
        data = preprocessor.transform(input_data)
        prediction = model.predict(data)[0]
        
        # Display prediction result
        st.header('Prediction Result')
        if prediction == 1:
            st.write("Customers comes under Cluster 1, have the following attributes:")
            st.write("- Higher income")
            st.write("- Higher amount spent")
            st.write("- Single or parent of less than 3 kids")
            st.write("- Higher amount of purchases")
            
        elif prediction == 0:
            st.write("Customer comes under Cluster 0, have the following attributes:")
            st.write("- Lower income")
            st.write("- Lower amount spent")
            st.write("- Married and parent of more than 3 kids")
            st.write("- Lower amount of purchases")

if __name__ == '__main__':
    main()
