# 🤖 Hugging Face Model Explorer

## Overview

A Streamlit application that allows users to interact with different language models from Hugging Face, including Mistral and GPT-2 models.

## System Architecture

![alt text](<Screenshot 2024-12-16 at 7.26.35 PM.png>)

## Features

- 🔑 API Token Authentication
- 🤖 Multiple Model Selection
- 🎛️ Dynamic Model Configuration
- 🔍 Step-by-Step Reasoning Approach

## Prerequisites

- Python 3.8+
- Hugging Face Account
- Hugging Face API Token

## Installation

#Clone the Repository

```bash
git clone https://github.com/ravix007/huggingface-model-explorer.git
cd huggingface-model-explorer
```

#Create a virtual environment and activate it

```bash
python3 -m venv venv
source venv/bin/activate

```

#Then install the dependencies

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Enter your Hugging Face API Token
3. Select a model
4. Adjust temperature and max length
5. Submit your query

## Supported Models

- Mistral-7B-Instruct-v0.2
- Mistral-7B-Instruct-v0.3
- GPT-2

## Configuration Options

- 🌡️ Temperature Control (0.0 - 1.0)
- 📏 Max Response Length (50 - 500 tokens)

## Error Handling

- API Token validation
- Model initialization error catching
- Graceful error messaging

## Screenshot of working application

![alt text](<Screenshot 2025-01-06 at 6.51.05 AM.png>)
