#!/usr/bin/env python
# coding: utf-8

# # Titanic Disaster Survival Using Logistic Regression

# In[31]:


#import libraries


# In[32]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ### Load the Data

# In[33]:


#load data


# In[34]:


titanic_data=pd.read_csv('train.csv')
len(titanic_data)


# In[35]:


titanic_data.head()


# In[36]:


titanic_data.index


# In[37]:


titanic_data.columns


# In[38]:



titanic_data.info()


# In[39]:


titanic_data.dtypes


# In[40]:


titanic_data.describe()


# ### Explaining Dataset
# #### survival : Survival 0 = No, 1 = Yes
# #### pclass : Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
# #### sex : Sex
# #### Age : Age in years
# #### sibsp : Number of siblings / spouses aboard the Titanic
# #### parch # of parents / children aboard the Titanic
# #### ticket : Ticket number fare Passenger fare cabin Cabin number
# #### embarked : Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

# # Data Analysis

# #### Import Seaborn for visually analysing the data

# #### Find out how many survived vs Died using countplot method of seaboarn

# In[41]:


#countplot of subrvived vs not  survived


# In[42]:


sns.countplot(x='Survived',data=titanic_data)


# #### Male vs Female Survival

# In[43]:


#Male vs Female Survived?


# In[44]:


sns.countplot(x='Survived',data=titanic_data,hue='Sex')


# #### See age group of passengeres travelled
# Note: We will use displot method to see the histogram. However some records does not have age hence the method will throw  an error. In order to avoid that we will use dropna method to eliminate null values from graph

# In[45]:


#Check for null


# In[46]:


titanic_data.isna()


# In[47]:


#Check how many values are null


# In[48]:


titanic_data.isna().sum()


# In[49]:


#Visualize null values


# In[50]:


sns.heatmap(titanic_data.isna())


# In[51]:


#find the % of null values in age column


# In[52]:


(titanic_data['Age'].isna().sum()/len(titanic_data['Age']))*100


# In[53]:


#find the % of null values in cabin column


# In[54]:


(titanic_data['Cabin'].isna().sum()/len(titanic_data['Cabin']))*100


# In[55]:


#find the distribution for the age column


# In[56]:


sns.displot(x='Age',data=titanic_data)


# # Data Cleaning

# #### Fill the missing values
# we will fill the missing values for age. In order to fill missing values we use fillna method.
# For now we will fill the missing age by taking average of all age
# 
# 

# In[57]:


#fill age column


# In[60]:


titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)


# #### We can verify that no more null data exist
# we will examine data by isnull mehtod which will return nothing

# In[ ]:


#verify null value


# In[62]:


titanic_data['Age'].isna().sum()


# #### Alternatively we will visulaise the null value using heatmap
# 
# we will use heatmap method by passing only records which are null.

# In[63]:


# Visulaize null values


# In[64]:


sns.heatmap(titanic_data.isna())


# #### We can see cabin column has a number of null values, as such we can not use it for prediction. Hence we will drop it
# 

# In[ ]:


#Drop cabin column


# In[65]:


titanic_data.drop('Cabin', axis = 1, inplace = True )


# In[ ]:


#see the contents of the data


# In[66]:


titanic_data.head()


# #### Preaparing Data for Model
# No we will require to convert all non-numerical columns to numeric. Please note this is required for feeding data into model. Lets see which columns are non numeric info describe method

# In[ ]:


#Check for the non-numeric column


# In[67]:


titanic_data.info()


# In[68]:


titanic_data.dtypes


# #### We can see, Name, Sex, Ticket and Embarked are non-numerical.It seems Name,Embarked and Ticket number are not useful for Machine Learning Prediction hence we will eventually drop it. For Now we would convert Sex Column to dummies numerical values**

# In[ ]:


#convert sex column to numerical values


# In[71]:


gender = pd.get_dummies(titanic_data['Sex'], drop_first = True)


# In[72]:


titanic_data['Gender'] = gender


# In[73]:


titanic_data.columns


# In[74]:


titanic_data.head()


# In[ ]:


#drop the columns which are not required


# In[75]:


titanic_data.drop(['Name','Sex','Ticket','Embarked'], axis = 1, inplace = True)


# In[77]:


titanic_data.head()


# In[ ]:


#Seperate Dependent(x) and Independent(y) variables


# In[80]:


x=titanic_data[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender']]
y=titanic_data['Survived']


# In[83]:


y


# # Data Modelling
# 
# 

# #### Building Model using Logestic Regression
# #### Build the model

# In[ ]:


#import train test split method


# In[84]:


from sklearn.model_selection import train_test_split


# In[ ]:


#train test split


# In[90]:


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)


# In[ ]:


#import Logistic  Regression


# In[91]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


#Fit  Logistic Regression


# In[94]:


lr = LogisticRegression()


# In[95]:


lr.fit(x_train, y_train)


# In[ ]:





# In[ ]:


#predict


# In[120]:


predict=lr.predict(x_test)


# In[ ]:





# # Testing

# #### See how our model is performing

# In[121]:


#print confusion matrix 


# In[148]:


from sklearn.metrics import confusion_matrix


# In[149]:


confusion_matrix(y_test,predict)


# In[144]:


#import classification report


# In[141]:


from sklearn.metrics import classification_report


# In[142]:


print(classification_report(y_test,predict))


# #### Precision is fine considering Model Selected and Available Data. Accuracy can be increased by further using more features                     (which we dropped earlier) and/or by using other model

# Note:
# Precision : Precision is the ratio of correctly predicted positive observations to the total predicted positive observations
# Recall : Recall is the ratio of correctly predicted positive observations to the all observations in actual class F1 score - F1 Score is the weighted average of Precision and Recall.

# In[ ]:




