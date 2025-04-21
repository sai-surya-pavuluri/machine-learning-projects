import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing...")

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load and sample dataset
try:
    df = pd.read_csv("movies.csv")
    logging.info("✅ Dataset loaded successfully. Total rows: %d", len(df))
except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise e

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    return " ".join(tokens)

# filter the required columns for recommendation
required_columns = ["genres", "keywords", "overview", "title"]

df = df[required_columns]

df = df.dropna().reset_index(drop=True)

df['content'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['title']

logging.info("🧹 Cleaning text...")
df['content'] = df['content'].apply(preprocess_text)
logging.info("✅ Text cleaned.")

# Vectorization
logging.info("🔠 Vectorizing using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['content'])
logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

# Cosine similarity
logging.info("📐 Calculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("✅ Cosine similarity matrix generated.")

# Save everything
joblib.dump(df, 'df_cleaned.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(cosine_sim, 'cosine_sim.pkl')
logging.info("💾 Data saved to disk.")

logging.info("✅ Preprocessing complete.")