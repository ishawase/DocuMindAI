<p align="center">
  <h1 align="center">DocuMindAI</h1>
  <p align="center">
    Retrieval-Augmented Generation System for Document Question Answering
  </p>
</p>


---

## Overview

DocuMindAI is a Retrieval-Augmented Generation (RAG) system that allows users to interact with their documents using natural language. The system retrieves relevant sections from a PDF file and generates context-grounded answers using a local Large Language Model.

This project demonstrates core AI concepts used in modern LLM applications.

---

**Pipeline Flow:**

Document → Chunking → Embeddings → Vector Database → Retrieval → LLM → Answer

---

## Features

- PDF document ingestion  
- Text chunking for efficient processing  
- Semantic embeddings using Sentence Transformers  
- Vector similarity search with FAISS  
- Local LLM (FLAN-T5) for grounded responses  
- Reduced hallucination via context-based prompting  

---

## Technology Stack

| Component | Technology |
|----------|------------|
| Language Model | Google FLAN-T5 Base |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS |
| Framework | LangChain Community Modules |
| Language | Python |

---

## Installation

Install dependencies:

```bash
pip install langchain-community langchain-core langchain-text-splitters sentence-transformers transformers faiss-cpu pypdf
