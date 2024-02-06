#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


mpgdf = pd.read_csv(r"C:\Users\hp\Python exercise\Auto MPG APP\AutoMPGReg.csv")


# In[3]:


mpgdf


# In[4]:


# Convert horsepower into numeric
mpgdf.horsepower=pd.to_numeric(mpgdf.horsepower,errors="coerce")


# In[5]:


mpgdf.horsepower=mpgdf.horsepower.fillna(mpgdf.horsepower.median())


# In[6]:


y = mpgdf.mpg
X = mpgdf.drop(['carname', 'mpg'], axis=1)


# In[7]:


from sklearn.linear_model import LinearRegression


# In[10]:


regmodel = LinearRegression().fit(X,y)


# In[11]:


regmodel.score(X,y)


# In[12]:


regpredict = regmodel.predict(X)


# In[13]:


from sklearn.metrics import mean_squared_error


# In[14]:


np.sqrt(mean_squared_error(y, regpredict))


# In[15]:


# For Deployement, model needs to be saved as ".pkl" (pickle) file or ".sav" (joblib) library


# In[16]:


import joblib


# In[17]:


joblib.dump(regmodel, 'reg.sav')


# In[18]:


# This is to know where the file was saved to
import os
os.getcwd()


# In[21]:


# Delete this cell after installing or checking 
#  !pip install streamlit
# So I kept it as a comment


# In[ ]:




