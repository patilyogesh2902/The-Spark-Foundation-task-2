#!/usr/bin/env python
# coding: utf-8
### The sparks Foundation: Data Science and Business Analytics Internship
### Task 2: Prediction using unsupervised ML Problem
### From the given iris dataset predict the optimum number of clusters and represent it visually
### Author: Patil Yogesh Sanjay
### Step-1 Importing the data set
#In[1]:
# Importing relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
# In[5]:
# load the data
data=pd.read_csv("C:/Users/DELL/Desktop/IRIS.csv")
# To show the first five columns of data set
data.head() 
# In[6]:
# Removing the 'Id' column
data.drop('Id' , axis=1, inplace=True) 
data.head()
### Step -2 Exploring the data
# In[7]:
# no of rows & columns
data.shape
# In[8]:
# Summary of Statistics
data.describe()
# In[9]:
# To check the if null values are Present
data.info()
# In[10]:
data.Species.value_counts()
### step 3 : Using Elow method to find optimal number  of clusters
# In[11]:
# Find the optimal number of clusters for k means clustering
x=data.iloc[:,:-1].values
from sklearn.cluster import KMeans
WCSS=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',
                 max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1,11),WCSS)
plt.title('The ELOW Method')
plt.xlabel('WCSS')
plt.show()
# In[12]:
# Apply means to the dataset
kmeans=KMeans(n_clusters=3,
             max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)
y_kmeans
## step 4: visualize the test set result
# In[13]:
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],
           s=100,c='red',label='Iris-setosa')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],
           s=100,c='blue',label='Iris-versicolour')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],
           s=100,c='green',label='Iris-virginica')

# Plotting the centorids of the clusters
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
           s=100,c='yellow',label='Centroids')
plt.legend()
### step 5: data visualization
# In[14]:
data.corr()
# In[15]:
plt.figure(figsize=(12,5))
sns.heatmap(data.corr(),annot=True,cmap='BuPu')
# In[16]:
plt.figure(figsize=(8,6))
sns.boxplot(x="Species",y="SepalLengthCm",data=data)
# In[17]:
sns.pairplot(data.corr())
## Thank you
# In[ ]:




