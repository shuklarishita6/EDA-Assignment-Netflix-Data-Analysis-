import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Task 1: Data Overview

df = pd.read_csv('netflix_customer_churn.csv')

df.head()

df.isnull().sum()

df.nunique()


#  Task 2: Univariate Analysis

sns.histplot(df['age'], kde=True)
plt.show()

sns.histplot(df['watch_hours'], kde=True)
plt.show()

sns.histplot(df['monthly_fee'], kde=True)
plt.show()

sns.countplot(x='churned', data=df)
plt.show()

sns.countplot(x='subscription_type', data=df)
plt.show()

sns.countplot(x='gender', data=df)
plt.show()

sns.countplot(x='region', data=df)
plt.show()

sns.countplot(x='device', data=df)
plt.show()

sns.countplot(x='payment_method', data=df)
plt.show()

sns.countplot(x='favorite_genre', data=df)
plt.xticks(rotation=45)
plt.show()


# Task 3: Bivariate Analysis

df.groupby('subscription_type')[['watch_hours', 'monthly_fee']].mean().plot(kind='bar')
plt.show()

df.groupby('region')[['watch_hours', 'monthly_fee']].mean().plot(kind='bar')
plt.show()

df.groupby('device')[['watch_hours', 'monthly_fee']].mean().plot(kind='bar')
plt.show()

df.groupby('favorite_genre')['avg_watch_time_per_day'].mean().plot(kind='bar')
plt.xticks(rotation=45)
plt.show()

df.groupby('gender')['churned'].mean().plot(kind='bar')
plt.show()

df.groupby('region')['churned'].mean().plot(kind='bar')
plt.show()

df.groupby('subscription_type')['churned'].mean().plot(kind='bar')
plt.show()

df.groupby('payment_method')['churned'].mean().plot(kind='bar')
plt.show()


#  Task 4: Correlation Analysis

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()
