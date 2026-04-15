# ============================================
# CLUSTER INTERPRETATION AND VALIDATION
# ============================================

import os
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# ============================================
# PATHS
# ============================================

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_folder = os.path.join(base_path, "data", "processed")
output_folder = os.path.join(data_folder, "plots")

os.makedirs(output_folder, exist_ok=True)

# ============================================
# LOAD DATA
# ============================================

print("=" * 70)
print("CLUSTER INTERPRETATION AND VALIDATION")
print("=" * 70)

print("\n[1] Loading data...")

data = pd.read_csv(os.path.join(data_folder, "clustered_data.csv"))
tfidf = load_npz(os.path.join(data_folder, "tfidf_matrix.npz"))

print(f"    Loaded {len(data)} complaints")
print(f"    TF-IDF shape: {tfidf.shape}")
print(f"    Clusters: {sorted(data['cluster'].unique())}")

n_clusters = data['cluster'].nunique()
print(f"    Number of clusters: {n_clusters}")

# ============================================
# GET TOP TF-IDF TERMS PER CLUSTER
# ============================================

print("\n[2] Extracting top TF-IDF terms per cluster...")

vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
vectorizer.fit(data['clean_text'])
feature_names = vectorizer.get_feature_names_out()

cluster_top_terms = {}

for cluster_id in sorted(data['cluster'].unique()):
    cluster_mask = data['cluster'] == cluster_id
    cluster_tfidf = tfidf[cluster_mask].toarray()
    
    mean_tfidf = cluster_tfidf.mean(axis=0)
    top_indices = mean_tfidf.argsort()[-10:][::-1]
    top_terms = [feature_names[i] for i in top_indices]
    
    cluster_top_terms[cluster_id] = top_terms
    print(f"    Cluster {cluster_id}: {', '.join(top_terms[:5])}...")

# ============================================
# SAMPLE INSPECTION AND LABELING
# ============================================

print("\n[3] Analyzing samples and assigning labels...")

cluster_labels = {}
cluster_samples = {}

for cluster_id in sorted(data['cluster'].unique()):
    cluster_data = data[data['cluster'] == cluster_id]
    samples = cluster_data['consumer_complaint_narrative'].head(3).tolist()
    cluster_samples[cluster_id] = samples
    
    top_terms = cluster_top_terms[cluster_id]
    top_products = cluster_data['product'].value_counts().head(3).index.tolist()
    top_issues = cluster_data['issue'].value_counts().head(3).index.tolist()
    
    label = f"Cluster {cluster_id} Label"
    
    keywords_lower = [t.lower() for t in top_terms[:5]]
    
    if any(t in keywords_lower for t in ['mortgage', 'loan', 'payment', 'escrow', 'principal']):
        label = "Mortgage/Loan Payment Issues"
    elif any(t in keywords_lower for t in ['credit', 'card', 'charge', 'balance', 'limit']):
        label = "Credit Card Problems"
    elif any(t in keywords_lower for t in ['account', 'bank', 'fund', 'transfer', 'withdraw']):
        label = "Bank Account Issues"
    elif any(t in keywords_lower for t in ['report', 'bureau', 'score', 'inquiry', 'dispute']):
        label = "Credit Report Disputes"
    elif any(t in keywords_lower for t in ['fraud', 'unauthorized', 'scam', 'identity', 'theft']):
        label = "Fraud/Unauthorized Activity"
    else:
        label = f"Miscellaneous (Top: {top_terms[0]}, {top_terms[1]})"
    
    cluster_labels[cluster_id] = {
        'label': label,
        'top_terms': top_terms,
        'top_products': top_products,
        'top_issues': top_issues,
        'size': len(cluster_data)
    }

# ============================================
# DETAILED OUTPUT
# ============================================

print("\n" + "=" * 70)
print("CLUSTER ANALYSIS RESULTS")
print("=" * 70)

summary_lines = []

for cluster_id in sorted(data['cluster'].unique()):
    info = cluster_labels[cluster_id]
    samples = cluster_samples[cluster_id]
    
    print(f"\n{'=' * 70}")
    print(f"CLUSTER {cluster_id}: {info['label']}")
    print(f"{'=' * 70}")
    print(f"Size: {info['size']} complaints ({info['size']/len(data)*100:.1f}%)")
    
    print(f"\n  Top 10 Keywords:")
    for i, term in enumerate(info['top_terms'], 1):
        print(f"    {i:2}. {term}")
    
    print(f"\n  Top Products: {', '.join(info['top_products'])}")
    print(f"  Top Issues: {', '.join(info['top_issues'])}")
    
    print(f"\n  Sample Complaints:")
    for i, sample in enumerate(samples, 1):
        clean_sample = sample[:300].replace('\n', ' ').strip()
        print(f"    {i}. {clean_sample}...")
    
    summary_lines.append({
        'Cluster': cluster_id,
        'Label': info['label'],
        'Size': info['size'],
        'Percentage': f"{info['size']/len(data)*100:.1f}%",
        'Top Terms': ', '.join(info['top_terms'][:5]),
        'Top Product': info['top_products'][0],
        'Top Issue': info['top_issues'][0]
    })

# ============================================
# VALIDATION: CLUSTER VS ACTUAL CATEGORIES
# ============================================

print("\n" + "=" * 70)
print("VALIDATION: CLUSTER ALIGNMENT WITH ACTUAL CATEGORIES")
print("=" * 70)

print("\n[4] Cluster vs Product distribution:")

for cluster_id in sorted(data['cluster'].unique()):
    cluster_data = data[data['cluster'] == cluster_id]
    product_dist = cluster_data['product'].value_counts()
    
    print(f"\n  Cluster {cluster_id} ({cluster_labels[cluster_id]['label']}):")
    for product, count in product_dist.head(3).items():
        pct = count / len(cluster_data) * 100
        print(f"    - {product}: {count} ({pct:.1f}%)")

print("\n[5] Cluster vs Issue distribution:")

for cluster_id in sorted(data['cluster'].unique()):
    cluster_data = data[data['cluster'] == cluster_id]
    issue_dist = cluster_data['issue'].value_counts()
    
    print(f"\n  Cluster {cluster_id}:")
    for issue, count in issue_dist.head(3).items():
        pct = count / len(cluster_data) * 100
        print(f"    - {issue}: {count} ({pct:.1f}%)")

# ============================================
# SAVE RESULTS
# ============================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

summary_df = pd.DataFrame(summary_lines)
summary_path = os.path.join(data_folder, "cluster_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n  [OK] Cluster summary saved to: {summary_path}")

text_output = []
text_output.append("=" * 70)
text_output.append("CLUSTER INTERPRETATION REPORT")
text_output.append("=" * 70)
text_output.append(f"Total complaints: {len(data)}")
text_output.append(f"Number of clusters: {n_clusters}")
text_output.append("")

for cluster_id in sorted(data['cluster'].unique()):
    info = cluster_labels[cluster_id]
    text_output.append(f"\nCLUSTER {cluster_id}: {info['label']}")
    text_output.append(f"Size: {info['size']} ({info['size']/len(data)*100:.1f}%)")
    text_output.append(f"Top Terms: {', '.join(info['top_terms'][:10])}")
    text_output.append(f"Top Products: {', '.join(info['top_products'])}")
    text_output.append(f"Top Issues: {', '.join(info['top_issues'])}")

text_output.append("\n" + "=" * 70)
text_output.append("VALIDATION NOTES")
text_output.append("=" * 70)
text_output.append("""
- Compare cluster labels with top products/issues
- If clusters align well with products, clustering is meaningful
- If misalignment exists, consider adjusting n_clusters
- The dendrogram suggested 5 clusters; 4 was used
""")

txt_path = os.path.join(output_folder, "cluster_analysis_report.txt")
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(text_output))

print(f"  [OK] Text report saved to: {txt_path}")

print("\n" + "=" * 70)
print("CLUSTER INTERPRETATION COMPLETE")
print("=" * 70)