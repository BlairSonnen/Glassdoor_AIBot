# Glassdoor_AIBot (the_death_star Project 3)

## Project Overview

This project aims to develop an  AI chatbot that uses the **Glassdoor Job Reviews 2** [Glassdoor Job Reviews](https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews-2?resource=download) dataset to provide insightful, data-driven responses to users exploring potential employers or job roles. The chatbot helps users understand company cultures, identify pros and cons of specific jobs or companies, and make informed career decisions.

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

## Step-by-Step Guide: Generating Cleaned Glassdoor Dataset  
Using Glassdoor Job Reviews 2 dataset from Kaggle with `data_cleanup.ipynb`

### Step 1: Install Dependencies

In the first section of the notebook, there is a cell to install the `kagglehub` Python library. This line is commented out to allow rerunning the notebook without unnecessary installation. If the library is not installed, simply uncomment the line, run the cell, and then comment it out again.

```python
# Install kagglehub if you need it, just comment out the next line
# !pip install kagglehub
```

---

### Step 2: Download the Data and Store it in a DataFrame

The `glassdoor-job-reviews-2` dataset is approximately 3.56 GB with over **9.9 million reviews**. The download process may take time initially, but the file will be cached locally after the first download. The dataset is then loaded into a Pandas DataFrame to facilitate data cleaning.

---

### Step 3: Extract the Company/Firm Names

The dataset includes a `firm_link` column containing a URL-safe string for each company's Glassdoor page. Although the dataset lacks a direct `company name` column, a helper function extracts the name from the `firm_link` and stores it in a new column, `firm_name`, resulting in **34,369 unique company names**.

---

### Step 4: Data Cleanup

Cleaning involves removing irrelevant or mostly-null columns such as `advice` and `index`. Rows containing any null entries are also dropped. Even after cleanup, millions of rows remain for analysis.

---

### Step 5: Bias Control

Initial chatbot use revealed a bias toward companies with a higher number of reviews. To mitigate this:
- Companies must have at least 1,000 reviews to be included.
- From each qualifying company, a random sample of 1,000 reviews is selected.

The result is saved to `Resources/cleaned_glassdoor_reviews_max1000.csv`, containing:
- **581,000 reviews**  
- **581 unique companies**

---

### Step 6: Model Consumption Management

Because consuming the cleaned dataset was time-intensive, two strategies were used to improve efficiency:

1. **Dataset Chunking**  
   The full dataset was split into smaller files, each with ≤ 25,000 rows. These are saved to `Resources/chunked/`.

2. **Reduced Sample File**  
   A smaller dataset was created by randomly sampling 500,000 rows from the full dataset. Companies with at least 500 reviews were kept, and a cap of 500 reviews per company was applied for bias control.  
   Final dataset saved as:  
   `Resources/cleaned_glassdoor_reviews_reduced_max500.csv`  
   Contains:
   - **131,000 reviews**  
   - **262 unique companies**

## Step-by-Step Guide: Running Your AI Code in Google Colab  
With Hugging Face Gated Model Access & Google Drive Integration

### Step 1: Setting Up Google Colab
1. Open Google Colab:
   
    - https://colab.research.google.com
  
2. Ensure that you are signed in with your Google account.

### Step 2: Open Project Notebook

1. Open your project notebook `Glassdoor_Ai_chatbot V1.1.ipynb` from:
   - The **Recent** tab if you’ve used it before
   - The **Google Drive** tab to browse your saved notebooks
   - Or upload it directly from your local computer

### Step 3: Install Dependencies
1. Install Hugging Face Transformers Library:
   
     - In a new cell, install the Hugging Face transformers library:
  ```
	!pip install huggingface_hub

	!huggingface-cli login
  ```
### Step 4: Accessing Hugging Face Gated Models
1. Open Hugging Face:
   
    - https://huggingface.co
   
2. Authenticate with Hugging Face:

    - If you need access to gated models, you must authenticate using your Hugging Face account token. First, create an account on Hugging Face if you don’t have one. 

### Step 5: Hugging Face account settings
1. Gated Repository settings:
   
    - On the top search bar, type the name of the module: [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
   
    - Install the model and then go to your account settings.
 Gated Repositories on the left side will show you the  installed module with the Accepted request status.
			
### Step 6: Create Hugging Face token
1. Access Token:
   
    - On the Hugging Face account settings page, go to the [Access Token](https://huggingface.co/settings/tokens) option. Click on Create new Token. 

    - Type a name for the new token. 

    - Under "Repositories", enable:
      - Read access to contents of all repos under your personal namespace
      - Read access to contents of all public gated repos you can access

    - Click **Create Token** and copy the token immediately  
    > Note: You cannot view the token again after clicking "Done"

### Step 7: Hugging Face CLI login
1. To log in with Hugging Face token:
   
    - Then run the command !huggingface-cli login (in Step 2), the Google Colab asks you to Enter your token (input will not be visible):
 
    - Paste the copied Hugging Face token and click the Enter button.
 
    - Add token as git credential? (Y/n): Type Y.

### Step 8: Mount Google Drive:
1. Write the Code to Mount Google Drive:
   
     - In the new code cell, you need to input the code to mount Google Drive. You can use the following code:
  ```
	from google.colab import drive

	drive.mount('/content/drive')
  ```
2. Run the Code Cell:
   
     - Click the "Run" button (play icon) on the left side of the cell, or press Shift + Enter to execute the code.

3. Authorize Access:
   
     - After running the cell, you will see a link in the output that says something like "Go to this URL in a browser:". Click on the link.
  
     - This action will redirect you to the Google account authentication page. Choose your Google account and allow access.
  
     - You'll then receive a verification code.

4. Paste the Authorization Code:
   
     - Copy the verification code and go back to your Colab notebook.

     - Paste the code into the prompt that appears in the output of the code cell and hit Enter.

5. Access Files:
   
     - Your Google Drive is now mounted and can be accessed at /content/drive. You can use standard file operations (like reading, writing, and navigating folders) just as you would with local files.

Example of Accessing Files:

To check if the drive has been mounted successfully and to list the files in your Google Drive, you can run the following command in a new code cell:

	!ls /content/drive/MyDrive/

This will display the contents of your "My Drive" folder in Google Drive.

### Step 9: Set File Paths and Load ChromaDB
Define your Chroma path:

CHROMA_PATH = "/content/drive/MyDrive/AI Chatbot Data/Glassdoor Chroma Store/chroma.sqlite3"

Use PersistentClient from ChromaDB to connect and load your collection.

### Step 10: Run the Model and App Code
1. Locate the cell beginning with # ai_recruiter_with_rag.py
   
2. Run your AI assistant logic (LLM loading, retrieval, inference, etc.)
   
3. Ensure paths and model identifiers are correctly configured.

### Step 11: Disconnect Google Drive
1. Unmount the Drive (optional):
   
     - If you wish to unmount your Google Drive after your session:
  ```
	drive.flush_and_unmount()
  ```

## AI chatbot - Glassdoor Job Reviews dataset 
Insert screenshot

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
Canonical URL  https://creativecommons.org/licenses/by-nc-sa/4.0/
