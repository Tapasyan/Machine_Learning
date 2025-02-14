#!/usr/bin/env python
# coding: utf-8

# # Sales Analysis Project
# ### Checking historical data
# ### For which category has the highest Sales
# ### Predict Future Sales On which Product of Furniture Sales Depend Most
# - Reading and understanding the data
# - Clean the data
# - handle unwanted Features
# - Convert Categorical Data into numerical one
# - Train XGBOOST Model

# In[146]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# In[235]:


furtinure = pd.read_csv(r"C:\\Users\\kumar\\OneDrive\\Desktop\\Furniture.csv")
furtinure.head()


# In[237]:


furtinure.info()


# In[239]:


furtinure.isnull().sum()


# # Here We don't have any Null Values in Our dataset

# In[242]:


furtinure.shape


# In[244]:


furtinure.describe()


# # We Don't have any Outlier in Our dataset

# In[247]:


furtinure["category"].value_counts()


# In[249]:


furtinure["material"].value_counts()


# In[251]:


furtinure["color"].value_counts()


# In[253]:


furtinure["location"].value_counts()


# In[255]:


furtinure["season"].value_counts()


# In[257]:


furtinure["brand"].value_counts()


# ## Drop unwanted features

# In[260]:


furtinure = furtinure.drop(["inventory", "discount_percentage" , "delivery_days"],axis = 1)


# ## Begain With Exploratory Data Analysis

# ## First For sales with categorical  Feature

# In[264]:


plt.figure(figsize=(5,3))
furtinure.groupby("category")["sales"].sum().plot.bar(color = 'cyan' , edgecolor = "black")
plt.show()


# #### All 5 category perform almost same but Table Sales slightly more with compare to other Four category

# In[267]:


plt.figure(figsize=(5,3))
furtinure.groupby("material")["sales"].sum().plot.bar(color = 'cyan' , edgecolor = "black")
plt.show()


# #### All 5 perform same just Metal and Wood slightly high sales

# In[270]:


plt.figure(figsize=(5,3))
furtinure.groupby("location")["sales"].sum().plot.bar(color = 'cyan' , edgecolor = "black")
plt.show()


# #### Rural Area has high sales compare to suburban and urban

# In[273]:


plt.figure(figsize=(5,3))
furtinure.groupby("season")["sales"].sum().plot.bar(color = 'cyan' , edgecolor = "black")
plt.show()


# #### In Winter season sales incrase as compare with other seasons

# In[276]:


furtinure.columns


# # Revenue VS Categorical Feature

# In[279]:


plt.figure(figsize=(5,4))
furtinure.groupby("category")["revenue"].sum().plot.pie(autopct= "%1.1f%%")
plt.show()


# In[281]:


plt.figure(figsize=(5,4))
furtinure.groupby("material")["revenue"].mean().plot.pie(autopct= "%1.1f%%")
plt.show()


# In[283]:


plt.figure(figsize=(5,4))
furtinure.groupby("location")["revenue"].mean().plot.pie(autopct= "%1.1f%%")
plt.show()


# ## Finding Pattern is there any pattern in Sales with respect to Season

# In[286]:


plt.figure(figsize=(5,4))
plt.plot(furtinure.groupby("season")["sales"].sum(), marker='*', linestyle='-')
plt.xlabel("Season")
plt.ylabel("Total Sales")
plt.title("Sales Trends Across Seasons")
plt.grid()
plt.show()


# ## Convert Categorical into numerical one

# In[289]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
categorical_cols = ["category", "material", "location", "season", "store_type", "brand", "color"]

for col in categorical_cols:
    furtinure[col] = encoder.fit_transform(furtinure[col])


# In[291]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.heatmap(furtinure.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()


# In[293]:


furtinure["unit_price"] = furtinure["revenue"] / furtinure["sales"]


# In[307]:


get_ipython().system('pip install xgboost')


# In[309]:


from xgboost import XGBRegressor
from sklearn.metrics import r2_score

xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
print("New R² Score:", r2_score(y_test, y_pred))


# In[311]:


import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')  # Perfect prediction line
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales (XGBoost)")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




