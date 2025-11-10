import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans

diamonds = sns.load_dataset("diamonds")

diamonds_numeric = diamonds.select_dtypes(include=[np.number])

def kmeans(X, k):
    X = np.array(X)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    return model.cluster_centers_, model.labels_

def kmeans_diamonds(n, k):
  
    X_subset = diamonds_numeric.head(n)
    return kmeans(X_subset, k)
