import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("bodyPerformance.csv")

features = ['sit-ups counts', 'broad jump_cm', 'sit and bend forward_cm']#TODO: add exercises to this list.
data2 = data[features]

kmeans = KMeans(n_clusters=3)
data2['clusters'] = kmeans.fit_predict(data2)

print(data2)
