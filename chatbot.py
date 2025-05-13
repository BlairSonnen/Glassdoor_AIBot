# ai_recruiter_with_rag.py (patched for Gradio file compatibility with full fallback support)

import subprocess
import sys
import importlib.util
import os

# Auto-install APT dependencies if in Colab
if os.path.exists("/content"):
    try:
        subprocess.check_call(["apt-get", "install", "-y", "libmagic1"])
    except Exception as e:
        print(f"⚠️ Failed to install apt dependency: {e}")

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
    "chromadb": "chromadb",
    "magic": "python-magic"
}

def install_missing(pkg_map):
    for imp_name, pip_cmd in pkg_map.items():
        if importlib.util.find_spec(imp_name) is None:
            print(f"📦 Installing {pip_cmd}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + pip_cmd.split())

install_missing(required)

import io
import time
import torch
import pandas as pd
import mammoth
import docx
import fitz
import xlrd
import magic
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

CHROMA_PATH = "/content/drive/MyDrive/Glassdoor Chroma Store"
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("csv_documents")

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

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_file(file):
    if file is None:
        return ""

    try:
        if hasattr(file, "read"):
            bytes_data = file.read()
        elif hasattr(file, "value") and os.path.exists(file.value):
            with open(file.value, "rb") as f:
                bytes_data = f.read()
        elif hasattr(file, "name") and os.path.exists(file.name):
            with open(file.name, "rb") as f:
                bytes_data = f.read()
        else:
            raise ValueError("Unsupported file object or missing file path.")
    except Exception as e:
        print(f"❌ Failed to read uploaded file: {e}")
        return ""

    mime_type = magic.from_buffer(bytes_data, mime=True)
    stream = io.BytesIO(bytes_data)

    try:
        if mime_type == "text/plain":
            return bytes_data.decode("utf-8", errors="ignore")[:3000]
        elif mime_type == "application/pdf":
            with fitz.open(stream=stream, filetype="pdf") as doc:
                return "\n".join(page.get_text() for page in doc)[:3000]
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return "\n".join(p.text for p in docx.Document(stream).paragraphs)[:3000]
        elif mime_type == "application/msword":
            return mammoth.extract_raw_text(stream).value[:3000]
        elif mime_type == "text/csv":
            return pd.read_csv(StringIO(bytes_data.decode("utf-8", errors="ignore"))).to_string()[:3000]
        elif "excel" in mime_type:
            return pd.read_excel(stream).to_string()[:3000]
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return ""

    return ""

def qa_with_llm(file, user_option, base_prompt, question):
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

    prompt = f"{base_prompt}Question: {question}\n\nContext:\n{context[:1000]}\n\nAnswer:"

    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096, padding=True)
    inputs = {k: v.to(model.device) for k, v in tokenized.items()}
    print(f"🧠 Prompt token count: {inputs['input_ids'].shape[1]}")

    try:
        start = time.time()
        output = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
        end = time.time()
        print(f"⏱️ Response time: {end - start:.2f} sec")
    except Exception as e:
        print(f"❌ Model generation failed: {e}")
        return "⚠️ An error occurred while generating the answer. Please try with a different file or question."

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    final_answer = answer.split("Answer:")[-1].strip()

    return final_answer

app = gr.Interface(
        fn=qa_with_llm,
        inputs=[
            gr.File(label="Upload Document"),
            gr.Radio([
                "Summarize the content",
                "Check for grammar/spelling/formatting",
                "Analyze the data"
            ], label="Task"),
            gr.Textbox(label="Base Prompt", value="You are an unbiased recruiting analyst. Use the data below to answer the question clearly and professionally."),
            gr.Textbox(label="Ask a question", placeholder="What are the top-rated companies for software engineers?")
        ],
        outputs="text",
        title="AI Recruiter Assistant",
        description="Upload a job-related document and/or ask the AI for recommendations or summaries."
    )

app.launch(share=True, debug=True)
