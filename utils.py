# ===============================
# IMPORTS
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import joblib
import os
import numpy as np


from collections import Counter
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import plotly.express as px

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ===============================
# EDA FUNCTION
# ===============================

def apply_eda(df):

    fig_num = 1

    print("="*60)
    print("BASIC DATA INFO")
    print("="*60)

    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    # Bar Chart
    print(f"\nFigure {fig_num}: Countries per Continent (Bar Chart)")
    df["Continent"].value_counts().plot(kind="bar")
    plt.title("Countries per Continent")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    fig_num += 1

    # Pie Chart
    print(f"\nFigure {fig_num}: Countries per Continent (Pie Chart)")
    df["Continent"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90
    )
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
    fig_num += 1

    # Word Count
    print(f"\nFigure {fig_num}: Distribution of Anthem Word Count")
    df["Word_count"] = df["Anthem"].apply(lambda x: len(str(x).split()))
    df["Word_count"].hist(bins=30)
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    fig_num += 1

    # Boxplot
    print(f"\nFigure {fig_num}: Word Count by Continent")
    sns.boxplot(x="Continent", y="Word_count", data=df)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    fig_num += 1

    # WordCloud
    print(f"\nFigure {fig_num}: WordCloud of All Anthems")
    text_data = " ".join(df["Anthem"].astype(str))
    wc = WordCloud(
        background_color="white",
        width=1000,
        height=500,
        colormap="viridis"
    ).generate(text_data)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("\n✅ EDA Completed\n")


# ===============================
# TEXT PREPROCESSING
# ===============================

lm = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

custom_words = {
    "thy", "thee", "thou", "shall",
    "may", "let", "us", "upon",
    "o", "ye"
}

stop_words = stop_words.union(custom_words)
exclude = string.punctuation


def standard_text(text):

    if not isinstance(text, str):
        return ""

    text = text.translate(str.maketrans("", "", exclude))
    text = text.lower()

    tokens = []
    for word in text.split():
        if word not in stop_words:
            lemma = lm.lemmatize(word)
            tokens.append(lemma)

    return " ".join(tokens).strip()


def light_clean(text):

    if not isinstance(text, str):
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ===============================
# KMEANS PIPELINE
# ===============================

def train_kmeans_pipeline(df, n_clusters=4):

    print("=" * 60)
    print("KMEANS PIPELINE STARTED")
    print("=" * 60)

    # 1. Load embedding model
    model = SentenceTransformer("all-MiniLM-L12-v2")

    # 2. Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(df["Anthem"].tolist())

    # 3. Elbow Method
    inertia = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(embeddings)
        inertia.append(km.inertia_)

    plt.figure()
    plt.plot(range(1, 10), inertia, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()

    # 4. Silhouette Score
    print("\nSilhouette Scores:")
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        print(f"{k} => {score:.4f}")

    # 5. Final Training
    kmean = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmean.fit_predict(embeddings)

    # 6. t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)

    df["x"] = reduced[:, 0]
    df["y"] = reduced[:, 1]

    plt.figure(figsize=(10, 7))
    plt.scatter(df["x"], df["y"], c=df["cluster"])
    plt.colorbar()
    plt.title("t-SNE Visualization")
    plt.show()

    # 7. Cluster Interpretation
    print("\nTop Words Per Cluster:")
    for c in range(n_clusters):
        text = " ".join(df[df["cluster"] == c]["Anthem_clean"])
        words = text.split()
        print(f"\nCluster {c}")
        print(Counter(words).most_common(10))

    theme_map = {
        0: "War & Sacrifice",
        1: "Unity & People",
        2: "Geography & Homeland",
        3: "National Pride"
    }

    df["Theme"] = df["cluster"].map(theme_map)

    # 8. Plotly Scatter
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Theme",
        hover_name="Country",
        title="Semantic Themes of National Anthems"
    )
    fig.show()

    print("\n✅ KMeans Pipeline Completed\n")

    # Create models folder if not exists
    os.makedirs("models", exist_ok=True)

    # Save KMeans model
    joblib.dump(kmean, "models/model.pkl")

    # Save embeddings
    np.save("models/embeddings.npy", embeddings)

    # Save dataframe
    df.to_pickle("models/df.pkl")

    # Save sentence transformer model
    model.save("models/sentence_model")

    print("\n✅ Models Saved Successfully in /models folder")

    return df, model, kmean, embeddings

# ===============================
# SAVE MODEL
# ===============================

def load_saved_models():

    import joblib
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    kmean = joblib.load("models/model.pkl")
    embeddings = np.load("models/embeddings.npy")
    df = pd.read_pickle("models/df.pkl")

    model = SentenceTransformer("models/sentence_model")

    print("✅ Models Loaded Successfully")

    return df, model, kmean, embeddings
# ===============================
# SHOW PREDICTION
# ===============================

def show_prediction(model, kmean, embeddings, df):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    theme_map = {
        0: "War & Sacrifice",
        1: "Unity & People",
        2: "Geography & Homeland",
        3: "National Pride"
    }

    print("\n" + "=" * 60)
    print("NATIONAL ANTHEM ANALYZER (Type 'exit' to quit)")
    print("=" * 60)

    while True:

        print("\nEnter your anthem text (Press Enter twice to submit):")

        lines = []
        while True:
            line = input()

            if line.lower() == "exit":
                print("\nExiting Anthem Analyzer. 👋")
                return

            if line == "":
                break

            lines.append(line)

        text = " ".join(lines).strip()

        if not text:
            print("⚠ No input provided. Try again.")
            continue

        # Generate embedding
        embedding = model.encode([text])

        # Predict cluster
        cluster = kmean.predict(embedding)[0]
        theme = theme_map.get(cluster, "Unknown")

        # Compute similarity
        similarities = cosine_similarity(embedding, embeddings)[0]

        # Get Top 3 indices (highest similarity)
        top3_idx = np.argsort(similarities)[-3:][::-1]

        print("\n" + "=" * 60)
        print("ANALYSIS RESULT")
        print("=" * 60)
        print(f"Predicted Cluster : {cluster}")
        print(f"Predicted Theme   : {theme}")
        print("\nTop 3 Similar Countries:")

        for rank, idx in enumerate(top3_idx, start=1):
            country = df.iloc[idx]["Country"]
            score = similarities[idx]
            print(f"{rank}. {country}  (Similarity: {score:.3f})")

        print("=" * 60)