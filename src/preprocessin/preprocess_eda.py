# ============================================
# DOCUMENT CLUSTERING - EDA + PREPROCESSING
# ============================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

pd.set_option('display.max_columns', None)

# ============================================
# PATH SETUP
# ============================================

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(base_path, "data", "banks_subset.csv")

output_folder = os.path.join(base_path, "data", "processed")
plots_folder = os.path.join(output_folder, "plots")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)

# ============================================
# LOAD DATA
# ============================================

print("Loading dataset...")
print("Trying path:", data_path)

data = pd.read_csv(data_path, low_memory=False)

print("\nDataset loaded successfully!")

# ============================================
# BASIC EDA
# ============================================

print("\n================ DATA PREVIEW ================")
print(data.head())

print("\nDataset shape:", data.shape)

print("\n================ DATA INFO ================")
data.info()

# ============================================
# MISSING VALUES
# ============================================

print("\n================ MISSING VALUES ================")
print(data.isnull().sum())

# ============================================
# CLEANING
# ============================================

print("\nDropping useless columns...")

drop_cols = ['Unnamed: 0', 'sub_issue', 'consumer_disputed', 'tags']
data.drop(columns=[c for c in drop_cols if c in data.columns], inplace=True)

data.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", data.shape)

data = data[['consumer_complaint_narrative', 'product', 'issue']]

data = data.dropna(subset=['consumer_complaint_narrative'])
print("Shape after dropping null text:", data.shape)

# ============================================
# TEXT CLEANING
# ============================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print("\nCleaning text...")
data['clean_text'] = data['consumer_complaint_narrative'].apply(clean_text)

# ============================================
# TEXT LENGTH ANALYSIS
# ============================================

data['text_length'] = data['clean_text'].apply(len)

print("\nText Length Stats:")
print(data['text_length'].describe())

# Plot
plt.figure()
plt.hist(data['text_length'], bins=30)
plt.title("Text Length Distribution")
plt.savefig(os.path.join(plots_folder, "text_length.png"))
plt.close()

# ============================================
# CATEGORY ANALYSIS
# ============================================

print("\nTop Products:")
print(data['product'].value_counts())

print("\nTop Issues:")
print(data['issue'].value_counts())

# ============================================
# WORD FREQUENCY
# ============================================

words = " ".join(data['clean_text']).split()
common = Counter(words).most_common(20)

print("\nTop 20 Words:")
print(common)

w = [i[0] for i in common]
c = [i[1] for i in common]

plt.figure()
plt.bar(w, c)
plt.xticks(rotation=45)
plt.title("Top Words")
plt.savefig(os.path.join(plots_folder, "top_words.png"))
plt.close()

# ============================================
# TF-IDF
# ============================================

print("\nApplying TF-IDF...")

tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
X = tfidf.fit_transform(data['clean_text'])

print("TF-IDF shape:", X.shape)

# ============================================
# SAVE
# ============================================

cleaned_path = os.path.join(output_folder, "cleaned_data.csv")
tfidf_path = os.path.join(output_folder, "tfidf_matrix.npz")

data.to_csv(cleaned_path, index=False)
sparse.save_npz(tfidf_path, X)

print("\nSaved cleaned data at:", cleaned_path)
print("Saved TF-IDF at:", tfidf_path)

print("\n======================================")
print("EDA + PREPROCESSING COMPLETED ✅")
print("NEXT STEP → Agglomerative Clustering 🚀")
print("======================================")