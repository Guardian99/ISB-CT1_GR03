#!/usr/bin/env python
# coding: utf-8

# ## What is a Conda Environment?
# A Conda environment is an isolated, self-contained directory that contains all the necessary dependencies—such as specific versions of Python, libraries, and tools—needed for a particular project or use case. These environments are created and managed using Conda, an open-source package and environment management system, which is often used in data science and software development to manage dependencies easily.
# 
# A Conda environment has:
# 
# - Python version (or potentially a different programming language runtime)
# - Libraries required for the project (like pandas, numpy, scikit-learn, etc.)
# - Scripts or other tools required for the project.
# 
# ### Create a conda environment
# 
# - Create a conda envionment 
# 
# ```
# conda create -n mlops python=3.9
# ```
# 
# - List conda environments
# 
# ```
# conda env list
# ```
# 
# - Activate the environment
# 
# ```
# conda activate mlops
# ```
# 
# - Print conda environment information
# 
# ```
# conda info
# ```
# 
# - Deactivate the environment
# 
# ```
# conda deactivate
# 
# ```
# 
# 
# 
# 

# ### Installing and Configuring mlflow
# 
# - Install mlflow library
# ```
# pip install mlflow
# ```
# 
# - Start the mlflow server
# 
# ```
# mlflow server --host 127.0.0.1 --port 8080
# ```
# 

# ### Import required libraries

# In[14]:


import mlflow
from mlflow.models import infer_signature

import numpy as np
import pandas as pd
import requests
import base64
from getpass import getpass
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# In[3]:


cars_df = pd.read_csv("used_car.csv")


# In[4]:


x_columns = ['KM_Driven', 'Fuel_Type', 'age',
              'Transmission', 'Owner_Type', 'Seats',
              'make', 'mileage_new', 'engine_new', 'model',
              'power_new', 'Location']


# In[5]:


cars_df = cars_df[x_columns + ['Price']].dropna()


# In[6]:


cat_features = ['Fuel_Type',
                'Transmission', 'Owner_Type', 'model',
                'make', 'Location']


# In[7]:


num_features = list(set(x_columns) - set(cat_features))


# In[8]:


train, test = train_test_split(cars_df,
                               train_size = 0.8,
                               random_state = 100)


# In[9]:


train.shape


# In[10]:


test.shape


# In[11]:


train.to_parquet("train.parquet")


# In[12]:


test.to_parquet("test.parquet")


# ### Version Control train and test datasets on Github
# 
# Useful commands for github
# 
# ```
# git clone https://github.com/username/mlopsdemo.git
# cd mlopsdemo
# cp /path/to/your/mlopsdemo/ .
# git add train.parquet
# git commit -m "Initial commit of dataset"
# git push origin main
# ```
# 

# In[18]:


train_df = pd.read_parquet("https://raw.githubusercontent.com/manaranjanp/MLOpsDemo/main/datasets/train.parquet")


# In[19]:


train_df.head(5)


# In[21]:


test_df = pd.read_parquet("https://raw.githubusercontent.com/manaranjanp/MLOpsDemo/main/datasets/test.parquet")


# ### Create the ML Pipeline

# In[23]:


numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[('onehot', 
                                           OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),                  
        ('cat', categorical_transformer, cat_features),
    ])

params = { "n_estimators": 400,
           "max_depth": 4 }

xgb_regressor = GradientBoostingRegressor(**params)

reg = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', xgb_regressor)])           

reg.fit(train_df[x_columns], 
        train_df['Price'])

rmse = np.sqrt(mean_squared_error(test_df['Price'], 
                                  reg.predict(test_df[x_columns])))
r2 = r2_score(test_df['Price'], reg.predict(test_df[x_columns]))


# ### Track Experiments

# In[24]:


# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:9090")

# Create a new MLflow Experiment
mlflow.set_experiment("UserCarPrice Experiment V1")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("rmse", rmse)
    # Log the loss metric
    mlflow.log_metric("r2", r2)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Gradient Boosting Model")

    # Infer the model signature
    signature = infer_signature(train_df[x_columns], 
                                reg.predict(train_df[x_columns]))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=reg,
        artifact_path="usedcarmodel",
        signature=signature,
        input_example=train_df[x_columns],
        registered_model_name="userd_car_model",
    )


# In[25]:


model_info.model_uri

