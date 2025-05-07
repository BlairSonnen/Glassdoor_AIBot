# simple_rag_csv_ingestion.py

import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure required packages are installed
required_packages = [
    "sentence_transformers",
    "chromadb",
    "pandas",
    "tqdm"
]

for pkg in required_packages:
    install_if_missing(pkg)

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Mount Google Drive if in Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

# Set ChromaDB path to Google Drive for persistence
CHROMA_PATH = "/content/drive/MyDrive/Glassdoor Chroma Store"
Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

# Initialize embedding model and ChromaDB client
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("csv_documents")

# Ingest a single CSV file
def ingest_csv(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        text = df.to_string(index=False)
        return (text, str(filepath))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

# Process and store CSV files
def process_csv_files(folder_path):
    folder = Path(folder_path)
    csv_files = list(folder.glob("**/*.csv"))

    batch = []
    for filepath in tqdm(csv_files, desc="Ingesting CSV files"):
        result = ingest_csv(filepath)
        if result:
            batch.append(result)

    if batch:
        texts, metadatas = zip(*batch)
        embeddings = embed_model.encode(list(texts))
        collection.add(
            documents=list(texts),
            embeddings=[e.tolist() for e in embeddings],
            metadatas=[{"filename": m} for m in metadatas],
            ids=[str(i) for i in range(len(texts))]  # use index as unique ID
        )
        client.persist()

    print(f"Ingested {len(batch)} CSV files.")

# Entry point for Google Colab
if __name__ == "__main__":
    # Update the path below to your Google Drive CSV folder
    process_csv_files("/content/drive/MyDrive/AI Chatbot Data")
