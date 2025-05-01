#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data=pd.read_excel("C://Users//pooja//OneDrive//Desktop//Documents//IITM//DAL//DA24C011//Default Dataset.xlsx")
#saved the original csv file as excel and then uploaded it as a pandas dataframe
data.columns=['x','y']


# In[3]:


rows=2000
columns=2000
#original img was 1599x1599 pixels, so considered a number slighlty greater for the rows and columns
mat_str= np.zeros((rows,columns))


# In[4]:


def sparseMatrix(data,mat_str):
    for i in range(len(data['x'])):
        mat_str[data.iloc[i][0],data.iloc[i][1]]=1
    return(mat_str)

def cordMatrix(sparse_mat):
    data = []
    for i in range(sparse_mat.shape[0]):
        for j in range(sparse_mat.shape[1]):
            if sparse_mat[i, j] == 1:
                data.append({'x': i, 'y': j})
    
    new_data = pd.DataFrame(data, columns=['x', 'y'])
    return new_data
                


# In[5]:


sparse_mat=sparseMatrix(data,mat_str)


# In[6]:


#first rotation
first_rot_sparse=np.rot90(sparse_mat,k=3,axes=(0,1))
first_rot_data=cordMatrix(first_rot_sparse)
plt.figure(figsize=(5,4))
sns.scatterplot(x=first_rot_data['x'], y=first_rot_data['y'],s=14)
plt.show()


# In[7]:


#second rotation
second_rot_sparse=np.rot90(sparse_mat,k=2,axes=(0,1))
second_rot_data=cordMatrix(second_rot_sparse)
plt.figure(figsize=(4,5))
sns.scatterplot(x=second_rot_data['x'], y=second_rot_data['y'],s=14)
plt.show()

