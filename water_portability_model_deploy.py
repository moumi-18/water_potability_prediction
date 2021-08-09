#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import GradientBoostingClassifier

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


h2o_data = pd.read_csv(r'C:\Users\RONALD\Desktop\IMS-Classroom\Python Code\Resume Project - ML Algo\Water Quality\archive\water_potability.csv')


# In[3]:


h2o_data.isnull().sum()


# #### Treating Missing Values

# In[4]:


def treat_num_columns(col_list):
    for i in col_list:
        col_name = i
        
        if (h2o_data[col_name].isnull().any().any() == True):
            
            h2o_data[col_name].fillna(h2o_data[col_name].median(),inplace=True)
            


# In[5]:


h2o_data.columns


# In[6]:


num_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']


# In[7]:


treat_num_columns(num_cols)


# In[8]:


h2o_data.isnull().sum()


# In[9]:


h2o_data.columns


# In[10]:


X = h2o_data.drop(['Potability'], axis=1)
Y = h2o_data[['Potability']]


# In[11]:


X.head()


# In[12]:


Y.head()


# In[13]:


gbclassifier = GradientBoostingClassifier(learning_rate=0.001, max_depth=10,
                           max_features='sqrt', min_samples_leaf=3,
                           n_estimators=1000)
gbclassifier.fit(X, Y)


# In[14]:


pickle.dump(gbclassifier, open('model.pkl', 'wb'))


# In[15]:


model = pickle.load(open('model.pkl', 'rb'))


# In[16]:


print(model.predict([[8.316765884, 214.3733941, 22018.41744, 8.059332377, 356.8861356, 363.2665162, 18.4365245, 100.3416744, 4.628770537]]))

