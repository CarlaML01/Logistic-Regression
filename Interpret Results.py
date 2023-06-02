#!/usr/bin/env python
# coding: utf-8

# ### Interpreting Results of Logistic Regression
# 
# In this notebook, I will be getting some practice with interpreting the coefficients in logistic regression.  
# 
# The dataset contains four variables: `admit`, `gre`, `gpa`, and `prestige`:
# 
# * `admit` is a binary variable. It indicates whether or not a candidate was admitted into UCLA (admit = 1) our not (admit = 0).
# * `gre` is the GRE score. GRE stands for Graduate Record Examination.
# * `gpa` stands for Grade Point Average.
# * `prestige` is the prestige of an applicant alta mater (the school attended before applying), with 1 being the highest (highest prestige) and 4 as the lowest (not prestigious).
# 
# To start, let's read in the necessary libraries and data.

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("./admissions.csv")
df.head()


# There are a few different ways someone might choose to work with the `prestige` column in this dataset.  For this dataset, I want to allow for the change from prestige 1 to prestige 2 to allow a different acceptance rate than changing from prestige 3 to prestige 4.
# 
# 1. With the above idea in place, let's create the dummy variables needed to change prestige to a categorical variable, rather than quantitative.

# In[2]:


# Creating dummy variables for prestige
dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')

# Joining the dummy variables to the original dataframe
df = pd.concat([df, dummy_ranks], axis=1)

# Drop the original prestige column
df.drop('prestige', axis=1, inplace=True)


# In[3]:


# Printing the first few rows of the updated dataframe
print(df.head())


# >This created three new columns in the dataframe, one for each of the three categories of prestige. The new columns were named 'prestige_2', 'prestige_3', and 'prestige_4', and contain binary values indicating whether or not the original prestige value was in that category.

# `2.` Now, let's fit a logistic regression model to predict if an individual is admitted using `gre`, `gpa`, and `prestige` with a baseline of the prestige value of `1`.  

# In[4]:


# Creating a baseline dummy variable for prestige 1
df['prestige_1'] = 0

# Defining the predictors and the response variable
X = df[['gre', 'gpa', 'prestige_2', 'prestige_3', 'prestige_4']]
y = df['admit']

# Adding an intercept term to the predictors
X = sm.add_constant(X)

# Fitting the logistic regression model
model = sm.Logit(y, X).fit()

# Printing the summary results
print(model.summary2())


# >All have p-values that suggest each is statistically significant.

# ### Interpretations:

# `1.` If an individual attended the most prestigious alma mater (prestige_1 as the baseline), they are more likely to be admitted than if they attended the least prestigious (prestige_4). 
#     - To calculate the odds ratio, we can exponentiate the coefficient of prestige_4: 
#     exp(-1.5534) = 0.211
#     - So, they are approximately 1/0.211 = 4.74 times more likely to be admitted.

# `2.` If an individual attended the most prestigious alma mater (prestige_1 as the baseline), they are more likely to be admitted than if they attended the second lowest in prestigious-ness (prestige_3). 
#     - To calculate the odds ratio, we can exponentiate the coefficient of prestige_3: exp(-1.3387) = 0.262. 
#     - So, they are approximately 1/0.262 = 3.82 times more likely to be admitted.

# `3.` If an individual attended the most prestigious alma mater (prestige_1 as the baseline), they are more likely to be admitted than if they attended the second most prestigious (prestige_2). 
#     - To calculate the odds ratio, we can exponentiate the coefficient of prestige_2: exp(-0.6801) = 0.506. 
#     - So, they are approximately 1/0.506 = 1.98 times more likely to be admitted.

# `4.` For every one point increase in GPA, an individual is more likely to be admitted, holding all other variables constant. 
#     - To calculate the odds ratio, we can exponentiate the coefficient of GPA: exp(0.7793) = 2.18. 
#     - So, for every one point increase in GPA, an individual is approximately 2.18 times more likely to be admitted.

# In[ ]:




