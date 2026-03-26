# 🤖 RAG Knowledge Assistant — InsuRelLM

> An end-to-end Retrieval-Augmented Generation (RAG) system that enables natural language Q&A over enterprise documents using vector embeddings, ChromaDB, and LLMs.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT-blueviolet)
![Gradio](https://img.shields.io/badge/Gradio-UI-red)

---

## 📌 Overview

This project builds a **production-grade RAG Knowledge Assistant** for **InsuRelLM**, a fictional insurance tech company. Employees can ask natural language questions about company data — employees, products, contracts, and company policies — and receive accurate, grounded answers powered by an LLM.

The project follows a **5-stage progressive build**, evolving from a simple keyword-based prototype to an advanced RAG system with semantic preprocessing, reranking, and query rewriting.

### Key Features

- 🔍 **Semantic Search** — Uses vector embeddings to find relevant documents by meaning, not just keywords
- 🧠 **Context-Aware Answers** — Injects retrieved documents into LLM prompts for grounded, accurate responses
- 📊 **76+ Document Knowledge Base** — Covers employees, products, contracts, and company information
- 💬 **Interactive Chat UI** — Gradio-powered chatbot for real-time querying
- 🔄 **Advanced RAG Techniques** — Semantic preprocessing, reranking, and query rewriting for production accuracy
- 🏠 **Local Model Support** — Ollama integration for fully offline operation (no API costs)

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────┐
│  Query Rewriter  │ ◄── Rewrites vague questions into precise search queries
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Search   │ ◄── ChromaDB with HuggingFace embeddings (384 dims)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Reranker      │ ◄── LLM re-orders results by true relevance
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Context Inject  │ ◄── Top chunks injected into system prompt
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Response   │ ◄── GPT generates a grounded answer
└────────┬────────┘
         │
         ▼
    Chat Interface (Gradio)
```

---

## 📂 Project Structure

```
RAG-Knowledge-Assistant-InsuRelLM/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── knowledge-base/                          # 📚 76 source documents
│   ├── employees/                           #   32 HR records
│   ├── products/                            #   8 product specs
│   ├── contracts/                           #   32 business contracts
│   └── company/                             #   4 company info docs
│
├── notebooks/                               # 📓 5-stage build journey
│   ├── 01_brute_force_rag.ipynb
│   ├── 02_chunking_and_vectorization.ipynb
│   ├── 03_rag_pipeline_and_chatbot.ipynb
│   ├── 04_evaluation_and_testing.ipynb
│   └── 05_advanced_rag.ipynb
│
├── app.py                                   # 🚀 Standalone chatbot app
├── evaluator.py                             # 📊 RAG evaluation framework
│
└── docs/                                    # 📝 Documentation
    └── summaries/
```

---

## 🗺️ The 5-Stage Build Journey

| Stage  | Notebook                              | What You Learn                                                                            |
| ------ | ------------------------------------- | ----------------------------------------------------------------------------------------- |
| **01** | `01_brute_force_rag.ipynb`            | Keyword-based RAG using Python dictionaries — the simplest possible approach              |
| **02** | `02_chunking_and_vectorization.ipynb` | Document chunking, vector embeddings (HuggingFace), ChromaDB storage, t-SNE visualization |
| **03** | `03_rag_pipeline_and_chatbot.ipynb`   | Full RAG pipeline with LangChain retriever + Gradio chat UI                               |
| **04** | `04_evaluation_and_testing.ipynb`     | Measuring RAG accuracy, comparing retrieval strategies                                    |
| **05** | `05_advanced_rag.ipynb`               | Semantic preprocessing, LLM reranking, query rewriting                                    |

---

## 🛠️ Tech Stack

| Category               | Technologies                      |
| ---------------------- | --------------------------------- |
| **LLM APIs**           | OpenAI (GPT), Ollama (Qwen/LLaMA) |
| **Embeddings**         | HuggingFace (`all-MiniLM-L6-v2`)  |
| **Vector Database**    | ChromaDB                          |
| **Framework**          | LangChain                         |
| **UI**                 | Gradio                            |
| **Visualization**      | Plotly, t-SNE (scikit-learn)      |
| **Structured Outputs** | Pydantic, LiteLLM                 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- OpenAI API key (for stages 1, 3, 4)
- Ollama installed locally (for stage 2 — optional, avoids API costs)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/RAG-Knowledge-Assistant-InsuRelLM.git
cd RAG-Knowledge-Assistant-InsuRelLM

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Run the Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order: 01 → 02 → 03 → 04 → 05
```

### Run the Standalone Chatbot

```bash
python app.py
# Opens in browser at http://127.0.0.1:7860
```

---

## 📊 Knowledge Base Details

The system indexes **76 markdown documents** across 4 categories:

| Category      | Count | Example Content                                                        |
| ------------- | ----- | ---------------------------------------------------------------------- |
| **Employees** | 32    | Name, title, salary, performance reviews, compensation history         |
| **Products**  | 8     | BizLLM, CarLLM, ClaimLLM, HealthLLM, HomeLLM, LifeLLM, MarkeLLM, ReLLM |
| **Contracts** | 32    | Client contracts with pricing, SLAs, and terms                         |
| **Company**   | 4     | About page, careers, culture, company overview                         |

---

## 📈 Results

| Approach                 | Accuracy | Limitations                                               |
| ------------------------ | -------- | --------------------------------------------------------- |
| **Brute-Force (Day 1)**  | Low      | Only works with exact keyword matches                     |
| **Vector RAG (Day 3)**   | Good     | Semantic search, but may retrieve irrelevant similar docs |
| **Advanced RAG (Day 5)** | Best     | Reranking + query rewriting eliminates most errors        |

---
