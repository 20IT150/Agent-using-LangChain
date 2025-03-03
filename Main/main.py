# app.py
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load documents and create vector store (FAISS for RAG)
docs = ["LangChain is an open-source framework for building LLM-powered applications.",
        "Retrieval-Augmented Generation (RAG) helps improve chatbot accuracy by fetching relevant context.",
        "Transformers power modern AI models, including ChatGPT and BERT."]

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(docs, embeddings)

# Create ConversationalRetrievalChain
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
retriever = vector_store.as_retriever()
qa_chain = ConversationalRetrievalChain.from_llm(chat_model, retriever)

def chat(query):
    response = qa_chain.run(query)
    return response

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print("Bot:", chat(user_input))

# prompt_engineering.py
class PromptEngineering:
    def __init__(self):
        self.base_prompt = "You are an AI assistant specialized in answering technical queries."

    def refine_prompt(self, user_query):
        return f"{self.base_prompt} Please answer the following question concisely: {user_query}"

# retrieval.py
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def create_vector_store():
    docs = ["LangChain is an open-source framework for building LLM-powered applications.",
            "Retrieval-Augmented Generation (RAG) helps improve chatbot accuracy by fetching relevant context."]
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(docs, embeddings)
    return vector_store

# requirements.txt
# Install dependencies
langchain
openai
faiss-cpu
python-dotenv

# README.md
# AI-Powered Conversational Agent using LangChain

## Overview
This project builds an AI chatbot using LangChain and OpenAI API with Retrieval-Augmented Generation (RAG) for enhanced response accuracy.

## Features
- Uses LangChain to interact with OpenAI’s LLMs
- Implements RAG with FAISS for better context retrieval
- Optimized prompt engineering for improved response generation

## Installation
```sh
pip install -r requirements.txt
```

## Usage
```sh
python app.py
```
