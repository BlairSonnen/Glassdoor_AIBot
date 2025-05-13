# Updated RAG Code

# simple_rag_csv_ingestion_chunked.py

import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

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
import gc

# Mount Google Drive if in Colab
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
except ImportError:
    pass

CHROMA_PATH = "/content/drive/MyDrive/Glassdoor Chroma Store"
Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("csv_documents")

def ingest_csv_chunked(filepath, chunk_size=50):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        total_rows = len(df)
        print(f"\n📄 Processing: {filepath.name} ({total_rows} rows)")

        batch_count = 0
        for start in tqdm(range(0, total_rows, chunk_size), desc=f"Chunks in {filepath.name}"):
            chunk = df.iloc[start:start + chunk_size]
            chunk_text = chunk.to_string(index=False)
            embedding = embed_model.encode(chunk_text)

            collection.add(
                documents=[chunk_text],
                embeddings=[embedding.tolist()],
                metadatas=[{"filename": str(filepath), "start_row": int(start)}],
                ids=[f"{filepath.stem}_{start}"]
            )

            batch_count += 1
            gc.collect()

        print(f"✅ Ingested {batch_count} chunks from {filepath.name}")
        return batch_count

    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return 0

def process_csv_files(folder_path, chunk_size=50):
    folder = Path(folder_path)
    csv_files = list(folder.glob("**/*.csv"))

    total_chunks = 0
    for filepath in csv_files:
        chunks = ingest_csv_chunked(filepath, chunk_size=chunk_size)
        total_chunks += chunks

    client.persist()
    print(f"\n✅ Done: Ingested {total_chunks} chunks total from {len(csv_files)} CSV file(s).")

if __name__ == "__main__":
    process_csv_files("/content/drive/MyDrive/AI Chatbot Data/Resources", chunk_size=50)
