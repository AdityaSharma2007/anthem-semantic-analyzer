# ===============================
# MAIN FILE - Anthem Analyzer
# ===============================

from utils import *
from setup_nltk import *

import os
import pandas as pd

# NLTK setup
nltk_call()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv("data/anthems.csv")

# ===============================
# EDA
# ===============================

apply_eda(df)

# ===============================
# PREPROCESSING
# ===============================

df["Anthem_clean"] = df["Anthem"].apply(standard_text)
df["Anthem_processed"] = df["Anthem"].apply(light_clean)

print("✅ Preprocessing Completed\n")

# ===============================
# TRAIN MODEL
# ===============================

df, model, kmean_model, embeddings = train_kmeans_pipeline(df)

# ===============================
# TEST PREDICTION
# ===============================

# Interactive prediction
show_prediction(
    model,
    kmean_model,
    embeddings,
    df
)



# anthem_analysis/
# │
# ├── data/
# │   └── anthems.csv
# │
# ├── models/
# │   ├── model.pkl
# │   ├── embeddings.npy
# │   ├── df.pkl
# │   └── sentence_model/
# │
# ├── main.py
# ├── utils.py
# ├── setup_nltk.py


# ===============================
# MODEL IS PRE-TRAINED
# ===============================

# import os
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
#
# from utils import load_saved_models, show_prediction
#
#
# def main():
#
#     print("=" * 60)
#     print("ANTHEM ANALYSIS SYSTEM (PRODUCTION MODE)")
#     print("=" * 60)
#
#     # -------------------------------
#     # Load Saved Models
#     # -------------------------------
#     try:
#         df, model, kmean_model, embeddings = load_saved_models()
#     except Exception as e:
#         print("❌ Error loading models.")
#         print("Make sure you trained and saved models first.")
#         print("Error:", e)
#         return
#
#     # -------------------------------
#     # Start Interactive Prediction
#     # -------------------------------
#     show_prediction(
#         model,
#         kmean_model,
#         embeddings,
#         df
#     )
#
#
# # ===============================
# # Run Program
# # ===============================
#
# if __name__ == "__main__":
#     main()
