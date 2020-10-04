#!/usr/bin/env python
# coding: utf-8

# # Importing  Libraries 

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing data from remote link

# In[2]:


url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("Data imported successfully")
df.head(10)


# # Plotting the distribution of scores

# In[3]:


df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage of Scores')  
plt.grid(alpha=0.3)
plt.show()


# # Training and Testing the data

# In[4]:


from sklearn.model_selection import train_test_split  
X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# # Shape of Training Data

# In[5]:


print(X_train.shape)
print(y_train.shape)


# # Import Linear Regression

# In[6]:


from sklearn.linear_model import LinearRegression  
lm = LinearRegression()  
lm.fit(X_train, y_train) 

print("Model Trained successfully.")


# # Plotting values and  regression line

# In[7]:


line = lm.coef_*X+lm.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,color='red')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage of Scores')  
plt.grid(alpha=0.3)
plt.show()


# # Printing the test data

# In[8]:


print(X_test) 
predictions = lm.predict(X_test) # Predicting the scores


# # Comparing Actual vs Predicted

# In[9]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})  
df 


# #  Marks Prediction when hours is 9.25

# In[10]:


hours = 9.25
pred_score = lm.predict([[hours]])
print("No of Hours = {} hrs".format(hours))
print("Predicted Score = {}".format(pred_score[0]))


# # Mean Absolute Error

# In[11]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, predictions)) 


# # Mean Squared Error

# In[12]:


from sklearn import metrics  
print('Mean Squared Error:', 
      metrics.mean_squared_error(y_test, predictions)) 


# # Root Mean Squared Error

# In[13]:


from sklearn import metrics  
print('Root Mean Squared Error:', 
     np.sqrt (metrics.mean_squared_error(y_test, predictions)) )

