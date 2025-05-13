# Glassdoor_AIBot (the_death_star Project 3)

## Project Overview

This project aims to develop an  AI chatbot that uses the **Glassdoor Job Reviews 2** dataset to provide insightful, data-driven responses to users exploring potential employers or job roles. The chatbot helps users understand company cultures, identify pros and cons of specific jobs or companies, and make informed career decisions.

## Purpose

To create a helpful, job-related AI interface that:
- Parses job review data from Glassdoor.
- Offers recommendations on companies or roles.
- Provides pros and cons based on real employee sentiment.
- Can be filtered by attributes like **location** (e.g., Utah), **job role** (e.g., System Administrator), and more.

## System Design

The system relies on a **Large Language Model (LLM)** prompted effectively to interpret and summarize data. The chatbot interface:
- Accepts natural language queries from the user.
- Leverages parsed company reviews and metadata (e.g., location, job title).
- Returns concise, conversational responses with relevant insights.

## How It Works

- A system prompt guides the LLM with the **persona**, **task**, **context**, and **format**.
- Users ask questions like:
  - *"What are some good companies for SysAdmin roles?"*
  - *"What are the pros and cons of working at Company X?"*
- The chatbot interprets the query, searches the dataset, and responds with structured summaries.

## Prompt Strategy

We must guide the LLM with effective instructions on:
- How to understand the structure and content of the Glassdoor dataset.
- How to summarize or compare company reviews.
- How to highlight patterns like common complaints, praise, or trends across locations or roles.

## Challenges 

The longer system prompts can consume space and use up the token limit. Why Long Prompts Consume Token Space.
1. Input Limitations:
  • The model has a fixed token limit (e.g., 4096 tokens for some models). This limit includes both the input (your prompt) and the output (the generated text). If the input is long, it leaves less room for the output.
2. Cost of Additional Tokens:
  • Every additional token in your system prompt reduces the number of tokens available for the output. This means that if your prompt is long, the model generates shorter responses because it has to stay within the maximum token limit.
Best Practices.
Concise prompts, essential information only, and iterative enhancement.

## Step-by-Step Guide: Running Your AI Code in Google Colab  
With Hugging Face Gated Model Access & Google Drive Integration

### Setup

### Step 1: Open Google Colab

1. Visit [https://colab.research.google.com](https://colab.research.google.com)
2. Open your project notebook `Glassdoor_Ai_chatbot V1.1.ipynb` from:
   - The **Recent** tab if you’ve used it before
   - The **Google Drive** tab to browse your saved notebooks
   - Or upload it directly from your local computer

---

### Step 2: Mount Google Drive via GUI

1. In the Colab interface, open the **Files** sidebar (click the folder icon on the left if hidden).
2. Click the **Mount Drive** button at the top of the sidebar.
3. A code cell will appear; run it:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Follow the authentication prompts and allow access.
5. Your Google Drive will now be accessible at `/content/drive/` (displayed as `drive/` in the file sidebar).

---

### Step 3: Install Required Python Packages

Run the following in a new cell if you need the Hugging Face library:

```bash
!pip install huggingface_hub
```

---

### Step 4: Get Access to a Gated Model on Hugging Face

1. Visit the desired model's page on Hugging Face (e.g., [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)).
2. Click **Agree and access repository** or **Request Access**.
3. Accept the terms and wait for approval.

---

### Step 5: Create Your Hugging Face Access Token

1. Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **Create New Token**
3. Name the token (e.g., `colab_token`) and set the role to **Read**
4. Under "Repositories", enable:
   - Read access to contents of all repos under your personal namespace
   - Read access to contents of all public gated repos you can access
5. Click **Create Token** and copy the token immediately  
   > Note: You cannot view the token again after clicking "Done"

---

### Step 6: Log in to Hugging Face from Colab

Use one of the following options:

**Option 1: Python**
```python
from huggingface_hub import login
login(token="YOUR_TOKEN_HERE")
```

**Option 2: CLI**
```bash
!huggingface-cli login
```
Paste your token into the prompt. When asked to add the token as a git credential, type `Y`.

---

### Step 7: Set File Paths and Load ChromaDB

Define your Chroma path:

```python
CHROMA_PATH = "/content/drive/MyDrive/AI Chatbot Data/Glassdoor Chroma Store/chroma.sqlite3"
```

Use `PersistentClient` from ChromaDB to connect and load your collection.

---

### Step 8: Run the Model and App Code

1. Locate the cell beginning with `# ai_recruiter_with_rag.py`
2. Run your AI assistant logic (LLM loading, retrieval, inference, etc.)
3. Ensure paths and model identifiers are correctly configured

---

### Optional Steps

#### Enable GPU

1. Go to **Runtime > Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Test with:
   ```python
   import torch
   torch.cuda.is_available()
   ```

#### Run RAG Ingestion

- Locate and run the large cell marked `# simple_rag_csv_ingestion.py`

#### Close the App

- The command `app.close()` is located in the last cell. Run it to cleanly shut down the interface.

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
Canonical URL  https://creativecommons.org/licenses/by-nc-sa/4.0/
