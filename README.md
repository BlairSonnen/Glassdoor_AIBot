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

## Challanges 

The longer system prompts can consume space and use up the token limit. Why Long Prompts Consume Token Space.
1. Input Limitations:
  • The model has a fixed token limit (e.g., 4096 tokens for some models). This limit includes both the input (your prompt) and the output (the generated text). If the input is long, it leaves less room for the output.
2. Cost of Additional Tokens:
  • Every additional token in your system prompt reduces the number of tokens available for the output. This means that if your prompt is long, the model generates shorter responses because it has to stay within the maximum token limit.
Best Practices.
Concise prompts, essential information only, and iterative enhancement.


