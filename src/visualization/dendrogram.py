# ============================================
# DENDROGRAM VISUALIZATION FOR HIERARCHICAL CLUSTERING
# ============================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import TruncatedSVD

# ============================================
# PATHS
# ============================================

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_folder = os.path.join(base_path, "data", "processed")
plots_folder = os.path.join(data_folder, "plots")

os.makedirs(plots_folder, exist_ok=True)

# ============================================
# LOAD TF-IDF MATRIX
# ============================================

print("=" * 60)
print("DENDROGRAM GENERATION - VERIFICATION")
print("=" * 60)

print("\n[STEP 1] Loading TF-IDF matrix...")
tfidf_path = os.path.join(data_folder, "tfidf_matrix.npz")

if not os.path.exists(tfidf_path):
    raise FileNotFoundError(f"TF-IDF matrix not found at: {tfidf_path}")

X_sparse = load_npz(tfidf_path)

print(f"  [OK] TF-IDF matrix loaded successfully")
print(f"  [OK] Shape: {X_sparse.shape}")
print(f"  [OK] Format: {X_sparse.format}")
print(f"  [OK] Dtype: {X_sparse.dtype}")
print(f"  [OK] Non-zero elements: {X_sparse.nnz}")

if X_sparse.shape[0] == 0 or X_sparse.shape[1] == 0:
    raise ValueError("TF-IDF matrix is empty")

# ============================================
# DIMENSIONALITY REDUCTION WITH TRUNCATEDSVD
# ============================================

print("\n[STEP 2] Applying TruncatedSVD...")

n_components = min(50, X_sparse.shape[0] - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(X_sparse)

print(f"  [OK] Reduced matrix shape: {X_reduced.shape}")
print(f"  [OK] Explained variance: {svd.explained_variance_ratio_.sum():.2%}")

if X_reduced.shape[0] == 0:
    raise ValueError("No samples after dimensionality reduction")

# ============================================
# HIERARCHICAL LINKAGE (WARD METHOD)
# ============================================

print("\n[STEP 3] Computing Ward linkage...")

Z = linkage(X_reduced, method='ward')

print(f"  [OK] Linkage matrix shape: {Z.shape}")
print(f"  [OK] Number of merges: {Z.shape[0]}")

if Z.shape[0] != X_reduced.shape[0] - 1:
    raise ValueError("Linkage matrix size mismatch")

# ============================================
# DENDROGRAM VISUALIZATION
# ============================================

print("\n[STEP 4] Generating dendrogram...")

fig, ax = plt.subplots(figsize=(14, 8))

dendrogram(
    Z,
    truncate_mode='level',
    p=5,
    leaf_rotation=90,
    leaf_font_size=9,
    show_contracted=True,
    ax=ax
)

ax.set_title('Hierarchical Clustering Dendrogram\n(Ward Linkage + TruncatedSVD)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Sample Index / (Cluster Size)', fontsize=12)
ax.set_ylabel('Distance (Ward)', fontsize=12)
ax.tick_params(axis='both', labelsize=10)

plt.tight_layout()

output_path = os.path.join(plots_folder, "dendrogram.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"  [OK] Dendrogram saved to: {output_path}")

# ============================================
# OPTIMAL CLUSTER ANALYSIS
# ============================================

print("\n[STEP 5] Cluster count suggestions...")

distances = Z[:, 2]
min_dist, max_dist = distances.min(), distances.max()

print(f"\n  Distance range: {min_dist:.2f} to {max_dist:.2f}")
print("\n  Suggested clusters at different thresholds:")

for pct in [0.3, 0.5, 0.7]:
    threshold = min_dist + (max_dist - min_dist) * pct
    n_clusters = len(np.unique(fcluster(Z, t=threshold, criterion='distance')))
    print(f"    - {int(pct*100)}% height ({threshold:.2f}): {n_clusters} clusters")

print("=" * 60)
print("DENDROGRAM GENERATION COMPLETE")
print("=" * 60)

print("""
EXPLANATION:
============
The dendrogram shows the hierarchical structure of document clusters.
- Y-axis: Distance (dissimilarity) between clusters
- X-axis: Individual documents or cluster merges
- Height of horizontal lines: Distance at which clusters merge

HOW TO INTERPRET:
=================
1. Tall vertical lines = distinct clusters with high separation
2. Short horizontal lines = similar documents merged together
3. The higher you cut horizontally, the fewer clusters you get

HOW TO CHOOSE OPTIMAL CLUSTERS:
===============================
1. Look for the tallest vertical line without horizontal crossing
2. Cut at a height where the vertical line is intersected
3. Count the number of vertical lines crossed = optimal clusters
4. Or use the printed distance thresholds above
""")