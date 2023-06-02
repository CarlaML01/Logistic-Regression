#!/usr/bin/env python
# coding: utf-8

# ### Fitting Logistic Regression
# 
# In this notebook, I will be fitting a logistic regression model to a dataset where I would like to predict if a transaction is fraud or not.
# 
# To get started let's read in the libraries and take a quick look at the dataset.

# In[15]:


import numpy as np
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('./fraud_dataset.csv')
df.head()


# `1.` As you can see, there are two columns that need to be changed to dummy variables.  Replacing each of the current columns to the dummy version.  Using the 1 for `weekday` and `True`, and 0 otherwise.  

# In[16]:


#Replacing the 'day' and 'fraud' columns with dummy variables.
#dummy variables

df['day'] = df['day'].replace({'weekday': 1, 'weekend': 0})
df['fraud'] = df['fraud'].replace({True: 1, False: 0})
df.info()


# In[18]:


#The proportion of fraudulent transactions:
df['fraud'].sum();
107/8793


# In[24]:


#The average duration for fraudulent transaction:
avg_duration_fraud = df[df['fraud'] == 1]['duration'].mean()

print("The average duration for fraudulent transactions is:", avg_duration_fraud)


# In[28]:


#The proportion of weekday transaction:
df['day'].sum();
3036/8793


# In[29]:


#The average duration for non-fraudulent transactions:
avg_duration_nfraud = df[df['fraud'] == 0]['duration'].mean()

print("The average duration for non-fraudulent transactions is:", avg_duration_nfraud)


# `2.` Now that I have dummy variables, fitting a logistic regression model to predict if a transaction is fraud using both day and duration.  Not forgeting an intercept!  

# In[30]:


#Adding an intercept column to the DataFrame.
df['intercept'] = 1


# In[31]:


# Defining the independent and dependent variables
X = df[['intercept', 'day', 'duration']]
y = df['fraud']

# Fitting a logistic regression model
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Displaying the summary of the fitted model
print(result.summary2())


# >Both duration and weekday had p-values suggesting they were statistically significant.

# > - On weekdays, the chance of fraud is 12.76 times more likely than on weekends holding duration constant.
#     - The exponentiated coefficient for 'day' is exp(2.5465) = 12.76. This means that the odds of fraud on weekdays are 12.76 times higher than on weekends, holding duration constant.
# > - For each minute less spent on the transaction, the chance of fraud is 4.32 times more likely holding the da of week constant.

# In[ ]:




