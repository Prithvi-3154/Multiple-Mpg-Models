#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[4]:


mpgdf=pd.read_csv("AutoMPGReg.csv")


# In[5]:


mpgdf.head()


# In[7]:


mpgdf.info()


# In[8]:


mpgdf.horsepower=pd.to_numeric(mpgdf.horsepower, errors="coerce")


# In[9]:


mpgdf.horsepower=mpgdf.horsepower.fillna(mpgdf.horsepower.median())


# In[10]:


# Split Data into independent variable and dependent variables
y=mpgdf.mpg
X=mpgdf.drop(['carname','mpg'], axis=1)


# In[11]:


# Defining Multiple Models as a "Dictionary"
models={'Linear Regression':LinearRegression(),'Decision Tree':DecisionTreeRegressor(),'Random Forest':RandomForestRegressor(),
        'Gradient Boosting':GradientBoostingRegressor()}


# In[12]:


# Side bar for Model Selection
selected_model=st.sidebar.selectbox("Select a ML model", list(models.keys()))
# The above line of code let's us selecting the model we want for result


# In[14]:


# ML model Selection Parameters
if selected_model=='Linear Regression':
    model=LinearRegression()
elif selected_model=='Decision Tree':
    max_depth=st.sidebar.slider("max_depth",8,16,2)
    model=DecisionTreeRegressor(max_depth=max_depth)
elif selected_model=='Random Forest':
    n_estimators=st.slider.sidebar("Num of Trees",100,500,50)  
    model=RandomForestRegressor(n_estimators=n_estimators)
elif selected_model=='Gradient Boosting':
    n_estimators=st.sidebar.slider("Num of Trees",100,500,50)
    model=GradientBoostingRegressor(n_estimators=n_estimators)

# in the brackets(starting count of the slider, max count of the slider, how many steps you want)



# In[15]:


# Train the model
model.fit(X,y)
# So by default it takes the Liner Regression


# In[17]:


# Define the Application Page Parameters
st.title("Predict Mileage per Gallon")
st.markdown("Model to Predict Mileage of a car")
st.header("Car Features")


# Now we are defining how many columns we want
col1,col2,col3,col4=st.columns(4)
with col1:
    cylinders=st.slider("Cylinders", 2,8,1)
# In the bracket(Slider lowest value, highest value,step size)
    displacement=st.slider("Displacement", 50,500,10)
with col2:
    horsepower=st.slider('Horse Power', 50,500,10)
    weight=st.slider("Weight",1500,6000,250)
with col3:
    acceleration=st.slider("Acceleration", 8,25,1)
    modelyear=st.slider("Year", 70,85,1)
with col4:
    origin=st.slider("Origin",1,3,1)


# In[18]:


# If you want the RSquare also, you can do the following
rsquare=model.score(X,y)
# Model Predictions
y_pred=model.predict(np.array([[cylinders,displacement,horsepower,weight,acceleration,modelyear,origin]]))


# In[19]:


# To Display results
st.header("ML Model Results")
st.write(f"Selected Model: {selected_model}")
st.write(f"Rsquare:{rsquare}")
st.write(f"Predicted:{y_pred}")
# Here "f" means formatting function, as the output needs to be displayed , hence represent output format 


# In[ ]:




