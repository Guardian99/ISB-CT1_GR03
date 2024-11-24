#!/usr/bin/env python
# coding: utf-8

# ### Import Required Libraries

# In[1]:


import mlflow
from mlflow.models import infer_signature

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ### Get the model from mlflow

# In[2]:


# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:9090")


# In[3]:


model_name = "userd_car_model"
model_version = "latest"

loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")


# In[4]:


loaded_model


# In[7]:


x_columns = ['KM_Driven', 'Fuel_Type', 'age',
              'Transmission', 'Owner_Type', 'Seats',
              'make', 'mileage_new', 'engine_new', 'model',
              'power_new', 'Location']


# In[6]:


test_df = pd.read_parquet("https://raw.githubusercontent.com/manaranjanp/MLOpsDemo/main/datasets/test.parquet")


# ### Load and Make Predictions

# In[8]:


predictions = loaded_model.predict(test_df[x_columns])


result = pd.DataFrame(test_df, columns=x_columns)
result["actual_price"] = test_df['Price']
result["predicted_price"] = predictions

result[:4]


# In[ ]:




