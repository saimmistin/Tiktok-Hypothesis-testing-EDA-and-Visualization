#!/usr/bin/env python
# coding: utf-8

# The purpose of this project is to conduct EDA on a TikTok data set. 
# Then and create visualizations and do hypothesis testing.

# In[1]:



# Import packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[2]:


# load dataset 
data = pd.read_csv("/Users/derya_ak/Desktop/tiktok.csv")


# In[3]:


# display and examine the first few rows 
data.head()


# In[4]:


#size of the data

data.size


# In[5]:


#shape of the data
data.shape


# In[6]:


#basic information about the data

data.info()


# In[7]:


#table of descriptive statistics

data.describe()


# In[8]:


# create a boxplot to visualize distribution of `video_duration_sec`
plt.figure(figsize=(5,1))
plt.title('video_duration_sec')
sns.boxplot(x=data['video_duration_sec']);


# In[9]:


pip install seaborn --upgrade


# In[13]:


#find the max duration sec
data['video_duration_sec'].max()


# In[14]:


# create a histogram
plt.figure(figsize=(5,3))
sns.histplot(data['video_duration_sec'], bins=range(0,61,5))
plt.title('Video duration histogram');


# In[15]:


# create a boxplot for  `video_view_count`
plt.figure(figsize=(5, 1))
plt.title('video_view_count')
sns.boxplot(x=data['video_view_count']);


# In[16]:


# create a histogram for further explore the distribution
plt.figure(figsize=(5,3))
sns.histplot(data['video_view_count'], bins=range(0,(10**6+1),10**5))
plt.title('Video view count histogram');


# In[17]:


# create a boxplot to visualize distribution of `video_like_count`
plt.figure(figsize=(10,1))
plt.title('video_like_count')
sns.boxplot(x=data['video_like_count']);


# In[18]:


# create a histogram
ax = sns.histplot(data['video_like_count'], bins=range(0,(7*10**5+1),10**5))
labels = [0] + [str(i) + 'k' for i in range(100, 701, 100)]
ax.set_xticks(range(0,7*10**5+1,10**5), labels=labels)
plt.title('Video like count histogram');


# In[19]:


# create a boxplot to visualize distribution of `video_comment_count`
plt.figure(figsize=(5,1))
plt.title('video_comment_count')
sns.boxplot(x=data['video_comment_count']);


# In[23]:


data['video_comment_count'].value_counts()


# In[24]:


# Create a histogram
plt.figure(figsize=(5,3))
sns.histplot(data['video_comment_count'], bins=range(0,(3500),100))
plt.title('Video comment count histogram');


# In[25]:


# create a boxplot to visualize distribution of `video_share_count`
plt.figure(figsize=(5,1))
plt.title('video_share_count')
sns.boxplot(x=data['video_share_count']);


# In[27]:


data['video_share_count'].max()


# In[28]:


#create a histogram
plt.figure(figsize=(5,3))
sns.histplot(data['video_share_count'], bins=range(0,(260000),10000))
plt.title('Video share count histogram');


# In[29]:


#create a boxplot to visualize distribution of `video_download_count`
plt.figure(figsize=(5,1))
plt.title('video_download_count')
sns.boxplot(x=data['video_download_count']);


# In[32]:


data['video_download_count'].max()


# In[33]:


# Create a histogram
plt.figure(figsize=(5,3))
sns.histplot(data['video_download_count'], bins=range(0,(15000),500))
plt.title('Video download count histogram');


# In[34]:


#claim status by verification status
#create a histogram with four bars:each combination of claim status and verification status.
plt.figure(figsize=(7,4))
sns.histplot(data=data,
             x='claim_status',
             hue='verified_status',
             multiple='dodge',
             shrink=0.9)
plt.title('Claims by verification status histogram');


# In[35]:


#claim status by author ban status
fig = plt.figure(figsize=(7,4))
sns.histplot(data, x='claim_status', hue='author_ban_status',
             multiple='dodge',
             hue_order=['active', 'under review', 'banned'],
             shrink=0.9,
             palette={'active':'green', 'under review':'orange', 'banned':'red'},
             alpha=0.5)
plt.title('Claim status by author ban status - counts');


# In[36]:


#median view counts by ban status


# In[38]:


ban_status_counts = data.groupby(['author_ban_status']).median(
    numeric_only=True).reset_index()

fig = plt.figure(figsize=(5,3))
sns.barplot(data=ban_status_counts,
            x='author_ban_status',
            y='video_view_count',
            order=['active', 'under review', 'banned'],
            palette={'active':'green', 'under review':'orange', 'banned':'red'},
            alpha=0.5)
plt.title('median view count by ban status');


# In[40]:


#calculate the median view count for claim status.
data.groupby('claim_status')['video_view_count'].median()


# In[41]:


#total views by claim status

fig = plt.figure(figsize=(3,3))
plt.pie(data.groupby('claim_status')['video_view_count'].sum(), labels=['claim', 'opinion'])
plt.title('Total views by video claim status');


# # hypothesis testing starts here

# In[42]:


# check for missing values
data.isna().sum()


# In[43]:


# drop rows with missing values
data = data.dropna(axis = 0)


# In[44]:


# compute the mean `video_view_count` for each group in `verified_status' verified 
#or not verified
data.groupby("verified_status")["video_view_count"].mean()


# Null hypothesis: There is no difference in the number of views between 
# tikTok videos' verified accounts and unverified accounts 
# 
# Alternative hypothesis: There is a difference in the number of views between 
# TikTok videos posted verified accounts and posted by unverified

# In[46]:


# conduct a two-sample t-test to compare means and 5% significant levels chosen
not_verified = data[data["verified_status"] == "not verified"]["video_view_count"]
verified = data[data["verified_status"] == "verified"]["video_view_count"]


# In[47]:


# implement a t-test using the two samples.
stats.ttest_ind(a=not_verified, b=verified, equal_var=False)


# Since the p-value is smaller (smaller than the significance level of 5%), we reject 
# the null hypothesis. 
# As a result, there is a statistically significant difference in the mean video view 
# count between verified and unverified accounts on TikTok.

# In[ ]:




