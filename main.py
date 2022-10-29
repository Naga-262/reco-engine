import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("bodyPerformance.csv")

features = ['sit-ups counts', 'broad jump_cm', 'sit and bend forward_cm']#TODO: add exercises to this list.
data2 = data[features]

kmeans = KMeans(n_clusters=3)
data2['clusters'] = kmeans.fit_predict(data2)

#The below code is just for plotting and helping us visualize the clusters
#pca_num_components = 2

#reduced_data = PCA(n_components=pca_num_components).fit_transform(data2)
#results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])

#sns.scatterplot(x='pca1', y='pca2', hue=data2['clusters'], data=results)
#plt.title('K-means Clustering')
#plt.show()
print(data2)
