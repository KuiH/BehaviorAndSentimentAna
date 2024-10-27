import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

SEED = 42
np.random.seed(SEED)
# 虚拟一些用户特征
STD = 11.5
CENTERS = 10  # 类别数
N_FEATURES = 40  # 维度

user_f, Y = make_blobs(n_samples=1200, n_features=N_FEATURES, cluster_std=STD, centers=CENTERS, random_state=SEED)
kmeans = KMeans(n_clusters=CENTERS, random_state=SEED)
Y_kmeans = kmeans.fit_predict(user_f)

tsne = TSNE(n_components=2, random_state=SEED)
X_tsne = tsne.fit_transform(user_f)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y_kmeans, cmap='viridis', alpha=0.8, edgecolors='w', s=50)

plt.title('K-Means Clustering with t-SNE Projection', fontsize=16)
plt.xlabel('t-SNE Component 1',fontsize=14)
plt.ylabel('t-SNE Component 2',fontsize=14)
plt.savefig(r'pics/user_cluster.png')
