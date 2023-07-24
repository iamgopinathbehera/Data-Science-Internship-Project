import pandas as pd
import numpy as np
import streamlit as st
import pickle

# streamlit run app.py--server.address=127.0.0.1
df = pd.read_csv('Model_df.csv')
ld_model = pickle.load(open('lr_model.pkl','rb'))
st.title('Profit Predictor')
st.header('Fill the Details for the Profit')

City = st.sidebar.selectbox('City',df['City'].unique())
Country = st.sidebar.selectbox('Country',df['Country'].unique())
Region = st.sidebar.selectbox('Region',df['Region'].unique())
Segment = st.sidebar.selectbox('Segment',df['Segment'].unique())
Ship_Mode = st.sidebar.selectbox('Ship Mode',df['Ship Mode'].unique())
State = st.sidebar.selectbox('State',df['State'].unique())
Days_to_Ship = st.sidebar.selectbox('Days to Ship',df['Days to Ship'].unique())
Product_Name = st.sidebar.selectbox('Product Name',df['Product Name'].unique())
Discount = st.sidebar.selectbox('Discount',df['Discount'].unique())
Actual_Discount  = st.sidebar.selectbox('Actual Discount',df['Actual Discount'].unique())
Sales = st.sidebar.selectbox('Sales',df['Sales'].unique())
Quantity = st.sidebar.selectbox('Quantity',df['Quantity'].unique())
Category = st.sidebar.selectbox('Category',df['Category'].unique())
Sub_Category = st.sidebar.selectbox('Sub-Category',df['Sub-Category'].unique())
Ship_Year = st.sidebar.selectbox('Ship_Year',df['Ship_Year'].unique())
Ship_Month  = st.sidebar.selectbox('Ship_Month',df['Ship_Month'].unique())
Ship_Day = st.sidebar.selectbox('Ship_Day',df['Ship_Day'].unique())
Ord_Year = st.sidebar.selectbox('Ord_Year',df['Ord_Year'].unique())
Ord_Month = st.sidebar.selectbox('Ord_Month',df['Ord_Month'].unique())
Ord_Day = st.sidebar.selectbox('Ord_Day',df['Ord_Day'].unique())



if st.button("Your Profit"):
      test_data = np.array([City, Country, Region, Segment, Ship_Mode, State, Days_to_Ship, Product_Name, Discount, Actual_Discount, Sales, Quantity, Category, Sub_Category, Ship_Year, Ship_Month, Ship_Day, Ord_Year, Ord_Month, Ord_Day])
      test_data = test_data.reshape([1,20])
      st.success(ld_model.predict(test_data)[0])