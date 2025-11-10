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


def kmeans_timer(n, k, n_iter=5):
   
    runtimes = []

    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        elapsed = time() - start
        runtimes.append(elapsed)

    return sum(runtimes) / len(runtimes)

