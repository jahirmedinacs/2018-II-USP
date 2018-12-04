
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris

from copy import copy as copy

import seaborn as sns


# In[2]:


def to_categorical(data_df, names_list):
    output = [None] * data_df.count()
    output = np.array(output)
    
    ref_idx = 0
    for ii in data_df.unique():
        temp_idx = data_df[data_df == ii].index.tolist()
        output[temp_idx] = names_list[ref_idx]
        
        ref_idx += 1
    
    return output


# In[3]:


def to_std_coord(data_df):
    out__ = (data_df - data_df.min()) / (data_df.max() - data_df.min())
    output = (out__ - out__.mean()) / out__.std()
    
    return output


# In[4]:


def pca__(data_set, verbose=True, convert=False):
    columns_name = data_set.columns.tolist()
    
    eig_vals, eig_vect = np.linalg.eig(data_set.cov().values)

    if convert:
        temp_df__ = np.dot(eig_vect.T, data_set.T.values)
        alt_output = pd.DataFrame(data=temp_df__.T, columns=columns_name)
    
    relv_ = pd.DataFrame(data=eig_vals)
    relv_.index = columns_name
    relv_.columns = ["relev"]
    relv_['%'] = ((relv_ / relv_.sum()) * 100).values
    if verbose:
        print("Relevance:")
        print(((relv_ / relv_.sum()) * 100))
    
    if convert:
        return eig_vals, eig_vect, relv_, alt_output
    else:
        return eig_vals, eig_vect, relv_


# In[5]:


iris = load_iris()


# In[6]:


iris.keys()


# In[7]:


iris_df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])


# In[8]:


for ii in range(len(iris['target_names'])):
    iris_df[iris['target_names'][ii]] = (iris['target'] == ii).astype(int)


# In[9]:


targets = iris_df.iloc[:, -3:]
iris_df = iris_df.iloc[:, :-3]


# In[10]:


iris_df = to_std_coord(iris_df)


# In[11]:


(eig_vals, eig_vect, by_relev, transf_df_) = pca__(iris_df, convert=True)


# In[12]:


# iris_pca = pd.DataFrame(data=np.dot(eig_vect.T, iris_df.T.values).T, columns=iris_df.columns.tolist())
iris_pca = transf_df_
iris_pca = to_std_coord(iris_pca)


# In[13]:


iris_pca['target'] = iris['target']
iris_pca['target'] = to_categorical(iris_pca['target'], iris['target_names'])


# In[14]:


df_to_plot = iris_pca

columns = df_to_plot.columns.tolist()

plt.figure(figsize=(20,20))
sns.pairplot(data=df_to_plot, vars=columns[:-1], hue='target')
plt.show()


# In[15]:


_2_most_relev = by_relev.sort_values(by=['%'], ascending=False)['%'].index.tolist()[:2]


# In[16]:


[pca_1, pca_2] = _2_most_relev
print(_2_most_relev)


# In[17]:


iris_pca[_2_most_relev].tail()


# In[18]:


df_to_plot = iris_pca[_2_most_relev + ['target']]
columns = df_to_plot.columns.tolist()

plt.figure(figsize=(20,20))
sns.pairplot(data=df_to_plot, vars=columns[:-1], hue='target')
plt.show()

