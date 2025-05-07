# ai_recruiter_with_rag.py

import subprocess
import sys
import importlib.util

# Auto-install and upgrade missing packages
required = {
    "torch": "torch",
    "pandas": "pandas",
    "mammoth": "mammoth",
    "docx": "python-docx",
    "fitz": "PyMuPDF",
    "xlrd": "xlrd",
    "sentence_transformers": "sentence-transformers",
    "transformers": "transformers --upgrade",
    "gradio": "gradio",
    "bitsandbytes": "git+https://github.com/TimDettmers/bitsandbytes.git",
    "accelerate": "accelerate --upgrade",
    "chromadb": "chromadb"
}

def install_missing(pkg_map):
    for imp_name, pip_cmd in pkg_map.items():
        if importlib.util.find_spec(imp_name) is None:
            print(f"📦 Installing {pip_cmd}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + pip_cmd.split())

install_missing(required)

import os
import io
import torch
import pandas as pd
import mammoth
import docx
import fitz
import xlrd
import numpy as np
from io import StringIO
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gradio as gr
from chromadb import PersistentClient

# Mount Google Drive if in Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

# Connect to ChromaDB in Google Drive
CHROMA_PATH = "/content/drive/MyDrive/AI Chatbot Data/Glassdoor Chroma Store"
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("csv_documents")

# Load model with fallback if bitsandbytes fails
model_path = "mistralai/Mistral-7B-Instruct-v0.2"
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    print("✅ Loaded model with 4-bit quantization (bnb)")
except Exception as e:
    print("⚠️ Failed to load with bitsandbytes, falling back to fp16")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

# Tokenizer and embedder
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Document loader

def extract_text_from_file(file):
    if file is None:
        return ""
    name = file.name
    ext = name.lower().split(".")[-1]
    content = ""
    bytes_data = file.read()

    if ext == "txt":
        content = bytes_data.decode("utf-8", errors="ignore")
    elif ext == "pdf":
        with fitz.open(stream=bytes_data, filetype="pdf") as doc:
            content = "\n".join(page.get_text() for page in doc)
    elif ext == "docx":
        doc = docx.Document(io.BytesIO(bytes_data))
        content = "\n".join([p.text for p in doc.paragraphs])
    elif ext == "csv":
        content = pd.read_csv(StringIO(bytes_data.decode("utf-8", errors="ignore"))).to_string()
    elif ext in ["xlsx", "xls"]:
        content = pd.read_excel(io.BytesIO(bytes_data)).to_string()
    elif ext == "doc":
        result = mammoth.extract_raw_text(io.BytesIO(bytes_data))
        content = result.value

    return content[:3000] if content else ""

# QA logic with RAG

def qa_with_llm(file, user_option, question):
    if file:
        context = extract_text_from_file(file)
    else:
        query_embedding = embed_model.encode(question).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        context = "\n\n".join(results['documents'][0]) if results and results['documents'] else ""

    if not question.strip():
        if not context:
            return "⚠️ Please upload a document or enter a question."
        question = f"Please {user_option.lower()} the following content:\n\n{context}"

    prompt = f"""You are a helpful AI recruiter assistant. Your task is to answer questions and give recommendations based on job data and candidate insights from Glassdoor.
{context[:1000]}
Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    final_answer = answer.split("Answer:")[-1].strip()

    return final_answer

# Gradio UI

def run_gradio():
    gr.Interface(
        fn=qa_with_llm,
        inputs=[
            gr.File(label="Upload Document"),
            gr.Radio([
                "Summarize the content",
                "Check for grammar/spelling/formatting",
                "Analyze the data"
            ], label="Task"),
            gr.Textbox(label="Ask a question", placeholder="What are the top-rated companies for software engineers?")
        ],
        outputs="text",
        title="AI Recruiter Assistant",
        description="Upload a job-related document and/or ask the AI for recommendations or summaries."
    ).launch(share=True, debug=True)

run_gradio()
