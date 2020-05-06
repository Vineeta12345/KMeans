# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:35:02 2020

@author: Vineeta
"""

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
%matplotlib inline
df = pd.read_csv("Kmeans.csv")
df.head()
plt.scatter(df['Production Budget'],df['Worldwide Gross Income'])
km = KMeans(n_clusters=3)
km
y_predicted = km.fit_predict(df[['Production Budget','Worldwide Gross Income']])
y_predicted
df['cluster'] = y_predicted
df.head()
df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]
plt.scatter(df0['Production Budget'],df0['Worldwide Gross Income'], color='red')
plt.scatter(df1['Production Budget'],df1['Worldwide Gross Income'], color='BLue')
plt.scatter(df2['Production Budget'],df2['Worldwide Gross Income'], color='yellow')
plt.xlabel('Production Budget')
plt.ylabel('Worldwide Gross Income')
plt.legend()
scaler = MinMaxScaler()
scaler.fit(df[['Worldwide Gross Income']])
scaler.fit(df[['Production Budget']])
df[['Worldwide Gross Income']] = scaler.transform(df[['Worldwide Gross Income']])
df
df[['Production Budget']] = scaler.transform(df[['Production Budget']])
km = KMeans(n_clusters=3)
km
y_predicted = km.fit_predict(df[['Production Budget','Worldwide Gross Income']])
y_predicted
df['cluster'] = y_predicted
df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]
plt.scatter(df0['Production Budget'],df0['Worldwide Gross Income'], color='red')
plt.scatter(df1['Production Budget'],df1['Worldwide Gross Income'], color='BLue')
plt.scatter(df2['Production Budget'],df2['Worldwide Gross Income'], color='yellow')
plt.xlabel('Production Budget')
plt.ylabel('Worldwide Gross Income')
plt.legend()
km.cluster_centers_
df0=df[df.cluster==0]
df1=df[df.cluster==1]
df2=df[df.cluster==2]
plt.scatter(df0['Production Budget'],df0['Worldwide Gross Income'], color='red')
plt.scatter(df1['Production Budget'],df1['Worldwide Gross Income'], color='BLue')
plt.scatter(df2['Production Budget'],df2['Worldwide Gross Income'], color='yellow')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
            color='green', marker='*', label='centroid')
plt.xlabel('Production Budget')
plt.ylabel('Worldwide Gross Income')
plt.legend()
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(df[['Production Budget','Worldwide Gross Income']])
    sse.append(km.inertia_)
    
sse
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)






