# 🌍 Global Semantic Analysis of National Anthems

## 📌 Project Overview

This project performs unsupervised semantic analysis of national anthems
using transformer-based embeddings and clustering techniques.

Instead of keyword matching or TF-IDF, this system leverages
Sentence-BERT (SBERT) embeddings to capture deep contextual meaning
and uncover global patriotic themes automatically.

Core Research Question:
Can we discover meaningful global anthem themes using purely unsupervised NLP methods?

---

# 🧠 Methodology

## 1️⃣ Text Representation (Semantic Embeddings)

Model Used:
- sentence-transformers (SBERT)

Why SBERT?
- Captures contextual semantic meaning
- Produces dense vector representations
- Works well with minimal preprocessing

Pipeline:
Anthem Text
    ↓
Basic Cleaning (lowercase, punctuation removal)
    ↓
SBERT Encoder
    ↓
Dense Embedding Vector (384 / 768 dimensions)

Key Insight:
Modern transformer embeddings require minimal preprocessing and
preserve semantic relationships across culturally different texts.

---

## 2️⃣ Clustering with KMeans

Goal:
Discover natural anthem groupings without labels.

Algorithm:
- KMeans clustering
- Optimal K selected using:
    - Elbow Method
    - Silhouette Score

Distance Metric:
- Euclidean distance in embedding space

Output:
Each anthem assigned to a semantic cluster.

Observed High-Level Clusters:
- War & Sacrifice
- National Pride
- Geography & Homeland
- Unity & People

---

## 3️⃣ Topic Modeling with BERTopic

Purpose:
Validate interpretability of KMeans clusters.

Why BERTopic?
- Uses transformer embeddings
- Applies dimensionality reduction (UMAP)
- Uses HDBSCAN for density-based clustering
- Extracts representative topic keywords

Pipeline:
Embeddings
    ↓
UMAP (dimensionality reduction)
    ↓
HDBSCAN (density clustering)
    ↓
c-TF-IDF (topic representation)

Outcome:
Thematic structures aligned closely with KMeans clusters,
increasing confidence in unsupervised results.

---

## 4️⃣ Similarity-Based Recommendation System

Using cosine similarity between anthem embeddings:

- Compute pairwise similarity matrix
- Retrieve Top-N most semantically similar anthems

This allows:
"Find anthems similar in patriotic narrative to Country X"

---

## 📊 Visualization

Techniques Used:

- t-SNE → 2D visualization of embedding clusters
- World Map (Plotly) → Geographic theme distribution
- Boxplots → Word count distribution by continent
- Scatter plots → Cluster comparison (KMeans vs BERTopic)

Purpose:
Enhance interpretability and storytelling of unsupervised patterns.

---

# 📈 Key Findings

1. Countries cluster semantically, not strictly geographically.
2. War-driven narratives appear across continents.
3. Unity and homeland themes form distinct embedding regions.
4. KMeans and BERTopic independently discovered similar thematic structures.
5. Transformer embeddings effectively capture cultural narrative tone.

---

# 🛠 Tech Stack

- Python
- Sentence-Transformers
- Scikit-learn
- BERTopic
- UMAP
- HDBSCAN
- Plotly
- Matplotlib
- Pandas
- NumPy

---

# 🚀 Project Significance

This project demonstrates:

- Practical application of modern NLP embeddings
- Unsupervised learning interpretability
- Model cross-validation without labeled data
- Cultural pattern discovery through semantic modeling
- End-to-end NLP pipeline design

---

# 🔮 Future Improvements

- Hierarchical clustering analysis
- Cross-lingual anthem embeddings
- Temporal evolution of patriotic themes
- Sentiment intensity analysis
- Graph-based similarity networks
- Interactive web dashboard deployment

---

# 📚 Learning Outcomes

- Deep understanding of transformer-based embeddings
- Comparative analysis of clustering techniques
- Importance of visualization in unsupervised learning
- Model agreement as a validation strategy
- Real-world cultural text analysis with AI
