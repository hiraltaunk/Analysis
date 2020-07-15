#!/usr/bin/env python
# coding: utf-8

# 
# 

# RFM
# To calculate recency, we need to find out most recent purchase date
# of each customer and see how many days they are inactive for. 
# After having no. of inactive days for each customer, we will apply K-means* clustering 
# to assign customers a recency score

# # Recency

# In[48]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#from __future__ import division
import chart_studio.plotly as py
import chart_studio.plotly as pyoff
import chart_studio.plotly as go


# In[7]:


df=pd.read_csv('OnlineRetail.csv',encoding='latin1')


# In[10]:


df.isnull().sum()


# In[21]:


df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'])
df_uk=df[df['Country']=='United Kingdom']


# In[22]:


df_uk.isnull().sum()


# In[27]:


#create a generic user dataframe to keep CustomerID and new segmentation scores
tx_user = pd.DataFrame(df_uk['CustomerID'].unique())
tx_user.columns = ['CustomerID']


# In[31]:


#get the max purchase date for each customer and create a dataframe with it
tx_max_purchase = df_uk.groupby('CustomerID').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']


# In[32]:


#we take our observation point as the max invoice date in our dataset
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days


# In[35]:


#merge this dataframe to our new user dataframe
tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')


# In[62]:


#plot a recency histogram

plt.subplots(figsize=(10,5))
plt.hist(tx_user['Recency'],bins=50)
plt.xlabel("Recency")
plt.show()


# In[51]:


tx_user.Recency.describe()


# ################################################################################################################################
# 
# We are going to apply K-means clustering to assign a recency score. But we should tell how many clusters we need to K-means algorithm. To find it out, we will apply Elbow Method. Elbow Method simply tells the optimal cluster number for optimal inertia

# In[66]:


from sklearn.cluster import KMeans
sse={}
tx_recency = tx_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_recency)
    tx_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()


# In[67]:


# To determine the optimal number of clusters, we have to select the value of k at the “elbow” 
#ie the point after which the distortion/inertia start decreasing in a linear fashion. 
#Thus for the given data, we conclude that the optimal number of clusters for the data is 3.
#Here it looks like 3 is the optimal one. Based on business requirements, we can go ahead with less or more clusters. 
#We will be selecting 4 for this example:


# In[89]:


#build 4 clusters for recency and add it to dataframe
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])
#clusters are
#0=48,130
#1=245,373
#2=0,47
#3=131,244


# In[81]:


#We have added one function to our code which is order_cluster(). 
#K-means assigns clusters as numbers but not in an ordered way. 
#We can’t say cluster 0 is the worst and cluster 4 is the best. 
#order_cluster() method does this for us and our new dataframe looks much neater:

def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)


#3 covers most recent customers whereas 0 has the most inactive ones.


# In[116]:


#see details of each Recency cluster
tx_user.groupby('RecencyCluster')['Recency'].describe()


# # Frequency

# In[92]:


tx_frequency = df_uk.groupby('CustomerID').InvoiceDate.count().reset_index()
tx_frequency.columns = ['CustomerID','Frequency']


# In[94]:


tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')


# In[96]:


tx_user


# In[109]:


plt.subplots(figsize=(10,5))
plt.hist(x=tx_user.query('Frequency < 1000')['Frequency'],bins=50)
plt.xlabel("Frequency")
plt.show()


# In[114]:


#K means to get the frequency cluster
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])

#order the frequency cluster
tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)


# In[115]:


#see details of each frequency cluster
tx_user.groupby('FrequencyCluster')['Frequency'].describe()


# # Revenue 

# In[117]:


df_uk['Revenue'] = df_uk['UnitPrice'] * df_uk['Quantity']
tx_revenue = df_uk.groupby('CustomerID').Revenue.sum().reset_index()


# In[118]:


#merge it with our main dataframe
tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')


# In[119]:


tx_user


# In[124]:


plt.subplots(figsize=(10,5))
plt.hist(x=tx_user.query('Revenue < 10000')['Revenue'],bins=50)
plt.xlabel("Revenue")
plt.show()


# In[125]:


#apply clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Revenue']])
tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])


#order the cluster numbers
tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)

#show details of the dataframe
tx_user.groupby('RevenueCluster')['Revenue'].describe()


# In[128]:


# Calculating overall score

tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
tx_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()


# In[129]:


#The scoring above clearly shows us that customers with score 8 is our best customers whereas 0 is the worst.

tx_user['Segment'] = 'Low-Value'
tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value'


# In[ ]:


#Things to do after finding insights 
#High Value: Improve Retention
#Mid Value: Improve Retention + Increase Frequency
#Low Value: Increase Frequency

#References:https://towardsdatascience.com/data-driven-growth-with-python-part-2-customer-segmentation-5c019d150444
