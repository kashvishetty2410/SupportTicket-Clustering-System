# ============================================
# DOCUMENT CLUSTERING - AGGLOMERATIVE
# ============================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.sparse import load_npz

# ============================================
# PATHS
# ============================================

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_folder = os.path.join(base_path, "data", "processed")
plots_folder = os.path.join(data_folder, "plots")

# ============================================
# LOAD DATA
# ============================================

print("Loading data...")

data = pd.read_csv(os.path.join(data_folder, "cleaned_data.csv"))
X = load_npz(os.path.join(data_folder, "tfidf_matrix.npz"))

print("Data loaded!")
print("TF-IDF shape:", X.shape)

X = X.toarray()

# ============================================
# CLUSTERING
# ============================================

print("\nApplying Agglomerative Clustering...")

model = AgglomerativeClustering(n_clusters=4)
clusters = model.fit_predict(X)

data['cluster'] = clusters

print("Clustering done!")

# ============================================
# CLUSTER DISTRIBUTION
# ============================================

print("\nCluster Distribution:")
print(data['cluster'].value_counts())

# ============================================
# SAMPLE OUTPUT
# ============================================

for i in range(4):
    print(f"\n===== CLUSTER {i} =====")
    print(data[data['cluster']==i]['consumer_complaint_narrative'].head(2))

# ============================================
# PCA VISUALIZATION
# ============================================

print("\nCreating PCA visualization...")

pca = PCA(n_components=2)
X_red = pca.fit_transform(X)

plt.figure()
plt.scatter(X_red[:,0], X_red[:,1], c=clusters)
plt.title("Cluster Visualization (PCA)")
plt.savefig(os.path.join(plots_folder, "cluster_plot.png"))
plt.close()

print("Cluster plot saved!")

# ============================================
# SAVE
# ============================================

output_path = os.path.join(data_folder, "clustered_data.csv")
data.to_csv(output_path, index=False)

print("\nSaved clustered data at:", output_path)

print("\n======================================")
print("CLUSTERING COMPLETED ✅")
print("======================================")