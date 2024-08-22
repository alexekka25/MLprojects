#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


import warnings

warnings.filterwarnings("ignore")


# # Student performance

# ### Life cycle of Machine Learning Project
# 
# 1. Understand the Problem statement
# 2. Data collection
# 3. Data checks to perform
# 4. Exploratory data analysis
# 5. model trainingl

# # 1) Problem Satement
#  
#  This project understand how the students performance (test score) is affected by others variables such as Gender, Ethnicty, Parental level of education, lunch and test prepration course.

# # 2) Data collection
# 
# .Dataset source - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977.

# # 3) Importing Data and Required Packages
# 
# - Pandas
# 
# - Numpy
# 
# - Matplotlib
# 
# - Seaborn
# 
# - Warnings

# In[25]:


df=pd.read_csv(r"D:\data analyst\datasets\archive\StudentsPerformance.csv")


# In[26]:


df.head()


# In[27]:


df.columns


# # 1: Understanding the Dataset
# 
# 1.Shape of the DataFrame
# 
# 2.Data Types
# 
# 3.Missing Values
# 
# 4.Basic Descriptive Statistics

# In[28]:


# Step 1: Overview of the dataset

# 1. Shape of the DataFrame
print("Shape of the dataset:", df.shape)

# 2. Data Types of each column
print("\nData Types:\n", df.dtypes)

# 3. Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# 4. Basic Descriptive Statistics for numerical columns
print("\nDescriptive Statistics:\n", df.describe())


# #  2: Univariate Analysis
# 
# 1.Categorical Variables: Frequency distribution of categories.
# 
# 2.Numerical Variables: Distribution and spread using histograms and box plots.

# In[ ]:





# ## 1. Frequency distribution of categorical variables

# In[29]:


# Step 2: Univariate Analysis

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Frequency distribution of categorical variables
categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, data=df)
    plt.title(f"Distribution of {col}")
    plt.show()



# ##  2. Distribution of numerical variables

# In[30]:


# 2. Distribution of numerical variables
numerical_columns = ['math score', 'reading score', 'writing score']

for col in numerical_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


# #  3: Bivariate Analysis
# 
# Explore relationships between pairs of variables:
# 
# 1.Categorical vs Numerical: Box plots or Violin plots to show distributions of numerical scores across categorical variables.
# 
# 2.Numerical vs Numerical: Scatter plots and correlation matrix to explore relationships.

# ## 1. Categorical vs Numerical: Box plots

# In[31]:


for cat_col in categorical_columns:
    for num_col in numerical_columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=cat_col, y=num_col, data=df)
        plt.title(f"{num_col} by {cat_col}")
        plt.show()



# In[ ]:





# ##  2. Numerical vs Numerical: Scatter plots and correlation matrix

# In[32]:


sns.pairplot(df[numerical_columns], diag_kind='kde')
plt.show()

correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(10, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# ## Insight
# 1.Strong Correlation Between Reading and Writing Scores: The correlation between reading and writing scores is extremely high at 0.95, indicating that students who score well in reading also tend to score well in writing.
# 
# 2.Moderate to Strong Correlation Between Math and Reading/Writing Scores: The math score has a strong positive correlation with both reading (0.82) and writing (0.80) scores, suggesting that students who perform well in math are also likely to do well in reading and writing, though the relationship is not as strong as between reading and writing.

# # 4: Multivariate Analysis
# Explore relationships between more than two variables:
# 
# 1.Pair Plots: Visualizing relationships for a combination of variables.
# 
# 2.Heatmaps: Visualizing correlations among multiple variables.

# In[34]:


# Step 4: Multivariate Analysis

# Pair Plots
sns.pairplot(df, hue='gender', diag_kind='kde')
plt.show()


# In[35]:


df['total score'] = df['math score'] + df['reading score'] + df['writing score']

# Calculate Average Score
df['average score'] = df['total score'] / 3

# Display the first few rows to check the new columns
df.head()


# In[43]:


reading_ful = df[df['reading score']==100]['average score'].count()
writing_full =df[df['writing score']==100]['average score'].count()
math_full = df[df['math score']==100]['average score'].count()


# In[47]:


print("Number of student with full reading marks",reading_ful)
print("Number of student with full writing marks",writing_full)
print("Number of student with full math marks",math_full)


# In[49]:


reading_less = df[df['reading score']<=20]['average score'].count()
writing_less =df[df['writing score']<=20]['average score'].count()
math_less = df[df['math score']<=20]['average score'].count()


# In[52]:


print("Number of student with less then 20 marks in reading",reading_less)
print("Number of student with  less then 20 marks in writing ",writing_less)
print("Number of student with less then 20 marks in math marks",math_less)


# In[54]:


sns.histplot(x=df['average score'],kde=True,hue=df['gender'])
plt.show()


# In[ ]:




