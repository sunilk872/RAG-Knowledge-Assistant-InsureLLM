# 📋 InsuRelLM RAG Knowledge Assistant — Detailed Project Summary

## 1. Problem Statement

Large Language Models (LLMs) are trained on general internet data and have **no knowledge of private, company-specific information** such as employee records, product details, or business contracts. When asked about such data, they either hallucinate (make up answers) or refuse to answer.

**RAG (Retrieval-Augmented Generation)** solves this by retrieving relevant documents from a private knowledge base and injecting them into the LLM's prompt at query time — giving the model accurate, up-to-date context without expensive fine-tuning.

This project builds a **complete RAG Knowledge Assistant** for **InsuRelLM**, a fictional insurance tech company, enabling employees to ask natural language questions and receive accurate, grounded answers from company data.

---

## 2. Knowledge Base

The system operates over **76 markdown documents** organized into 4 categories:

### 2.1 Employees (32 documents)
Each file is a full HR record containing:
- Name, date of birth, job title, location, salary
- Career progression at InsuRelLM
- Annual performance ratings (scale of 5) with commentary
- Full compensation history (base salary + bonus per year)
- Education, certifications, awards, and HR notes

**Example employees:** Avery Lancaster (CEO, $225K), Robert Chen (Sr. Full Stack Engineer, $152K), Alex Harper (SDR, $75K), and 29 others across engineering, HR, sales, and leadership.

### 2.2 Products (8 documents)
Each file describes an AI-powered insurance product:

| Product | Domain | Description |
|---------|--------|-------------|
| **BizLLM** | Business Insurance | AI-powered business insurance platform |
| **CarLLM** | Auto Insurance | AI risk assessment, instant quoting, fraud detection |
| **ClaimLLM** | Claims Processing | AI claims automation and management |
| **HealthLLM** | Health Insurance | Health insurance solutions with AI analytics |
| **HomeLLM** | Home Insurance | Home insurance portal with smart pricing |
| **LifeLLM** | Life Insurance | AI-driven life insurance product |
| **MarkeLLM** | Marketing | AI marketing engine for insurance |
| **ReLLM** | Reinsurance | Enterprise reinsurance solution |

Each product document includes: summary, features, pricing tiers (Basic/Professional/Enterprise), and a 2025-2026 roadmap.

### 2.3 Contracts (32 documents)
Business contracts between InsuRelLM products and client companies. Each contains:
- Client name and product used
- Contract terms, pricing, SLAs
- Implementation details

**Examples:** "Contract with DriveSmart Insurance for CarLLM", "Contract with SafeHaven Property Insurance for HomeLLM"

### 2.4 Company (4 documents)
- `about.md` — Company history and mission
- `careers.md` — Open positions and recruitment info
- `culture.md` — Company values and work culture
- `overview.md` — Business overview and market position

---

## 3. System Architecture

The project follows a **5-stage progressive build**, each stage adding a more sophisticated layer:

```
Stage 1 (Day 1)     Stage 2 (Day 2)      Stage 3 (Day 3)      Stage 5 (Day 5)
─────────────────    ─────────────────    ─────────────────    ─────────────────
Keyword Matching  →  Vector Embeddings →  Full RAG Pipeline →  Advanced RAG
(Dictionary)         (ChromaDB)           (LangChain+Gradio)   (Rerank+Rewrite)
```

### 3.1 Stage 1: Brute-Force Keyword RAG

**How it works:**
1. All 76 documents are loaded into a Python dictionary keyed by last name (employees) or product name (products)
2. When the user asks a question, the text is cleaned and split into words
3. Each word is checked against dictionary keys — if a match is found, the full document is retrieved
4. Retrieved documents are injected into the system prompt as context
5. The LLM generates an answer grounded in that context

**Retrieval function:**
```python
def get_relevant_context(message):
    text = ''.join(ch for ch in message if ch.isalpha() or ch.isspace())
    words = text.lower().split()
    return [knowledge[word] for word in words if word in knowledge]
```

**Limitations:**
- Only works if the user types an exact keyword (e.g., "Lancaster" finds data, but "Who is the CEO?" returns nothing)
- Cannot understand meaning, synonyms, or context
- No ranking — all matches are treated equally

### 3.2 Stage 2: Chunking & Vectorization

**How it works:**
1. **Document loading** — LangChain's `DirectoryLoader` reads all 76 `.md` files with metadata (doc_type: employees/products/contracts/company)
2. **Chunking** — `RecursiveCharacterTextSplitter` breaks documents into ~1000-character chunks with 200-character overlap, ensuring no information is lost at boundaries
3. **Embedding** — Each chunk is converted into a 384-dimensional vector using HuggingFace's `all-MiniLM-L6-v2` model (runs locally, no API cost)
4. **Storage** — Vectors are persisted to disk in a ChromaDB database (`vector_db/`)
5. **Visualization** — t-SNE reduces 384 dimensions to 2D for plotting with Plotly, revealing natural clusters by document type

**Key code:**
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="vector_db")
```

**Why 384 dimensions?** The `all-MiniLM-L6-v2` model maps any text into a 384-dimensional space where semantically similar texts are close together. This enables "search by meaning" instead of "search by keyword."

### 3.3 Stage 3: RAG Pipeline + Chatbot UI

**How it works:**
1. **Connect** — Load the ChromaDB vector store created in Stage 2
2. **Retrieve** — `vectorstore.as_retriever()` turns the database into a searchable tool that returns the most relevant chunks for any query
3. **Generate** — Retrieved chunks are injected into a system prompt and sent to `ChatOpenAI(temperature=0)` for factual, deterministic responses
4. **Deploy** — Gradio provides an interactive chat interface with conversation history

**RAG function:**
```python
def answer_question(question, history):
    # RETRIEVE: Find relevant document chunks
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # GENERATE: Ask LLM with injected context
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])
    return response.content
```

**Key insight:** RAG essentially "stuffs the model's short-term memory" (the prompt) with the right facts before it starts generating. The model doesn't learn anything new — it just reads the right documents at the right time.

### 3.4 Stage 4: Evaluation & Testing

**How it works:**
1. A test suite of questions with known answers is prepared
2. The RAG system is run against these questions
3. Metrics are collected: accuracy, relevance, hallucination rate
4. Results are compared across different retrieval approaches (keyword vs. vector vs. advanced)

**Key file:** `evaluator.py` — A custom evaluation framework that automates testing and scoring.

### 3.5 Stage 5: Advanced RAG (Production-Grade)

Three advanced techniques transform the basic RAG into a professional-grade system:

#### Technique 1: Semantic Preprocessing
Instead of blindly splitting text by character count, an **LLM reads each document** and creates structured chunks:

```python
class Chunk(BaseModel):
    headline: str    # A brief heading for this chunk
    summary: str     # A few sentences summarizing the content
    original_text: str  # The original text
```

The vector store now indexes `headline + summary + original_text`, producing significantly better embeddings than raw text alone.

#### Technique 2: Reranking
After vector search returns the top N results, a **secondary LLM re-orders them** by exact relevance to the question. This catches cases where semantically similar but irrelevant documents get surfaced by vector similarity alone.

```python
def rerank(question, chunks):
    # LLM evaluates: "Which of these actually answers the question?"
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    order = RankOrder.model_validate_json(response.choices[0].message.content).order
    return [chunks[i - 1] for i in order]
```

#### Technique 3: Query Rewriting
Converts vague, conversational follow-up questions into precise, standalone search queries using conversation history.

**Before:** "Who won it?"
**After:** "Winner of International Insurance Specialist of the Year 2024"

```python
def rewrite_query(question, history=[]):
    response = completion(model=MODEL, messages=[{"role": "system", "content": prompt}])
    return response.choices[0].message.content
```

#### Complete Advanced Pipeline:
```
User Question → Query Rewriting → Semantic Retrieval → Reranking → LLM Answer
```

---

## 4. Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **LLM (Cloud)** | OpenAI GPT | Answer generation, semantic preprocessing, reranking |
| **LLM (Local)** | Ollama (Qwen 2.5:3b) | Cost-free local alternative |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Converts text → 384-dim vectors (runs locally) |
| **Vector DB** | ChromaDB | Persistent local vector storage and similarity search |
| **Framework** | LangChain | Document loading, text splitting, retrieval orchestration |
| **UI** | Gradio | Interactive chat interface |
| **Structured Output** | Pydantic + LiteLLM | Enforces schema for semantic chunks and reranking |
| **Visualization** | Plotly + scikit-learn (t-SNE) | 2D/3D visualization of vector space |
| **Tokenization** | tiktoken | Token counting for context window management |
| **Environment** | python-dotenv | API key management |

---

## 5. Key Concepts Explained

### 5.1 What is RAG?
**Retrieval-Augmented Generation** is a technique where an LLM's prompt is dynamically augmented with relevant documents retrieved from a knowledge base. This gives the model access to private, up-to-date information without fine-tuning.

### 5.2 What are Vector Embeddings?
Embeddings are numerical representations of text in high-dimensional space (384 dimensions in this project). Texts with similar meaning have vectors that are close together, enabling "search by meaning."

### 5.3 What is ChromaDB?
ChromaDB is an open-source vector database that stores embeddings and supports similarity search. It persists to disk, so you only need to create the database once.

### 5.4 What is Chunking?
Large documents may exceed the LLM's context window or contain too much noise. Chunking splits them into smaller, focused pieces (1000 chars with 200-char overlap) so only the most relevant piece is retrieved.

### 5.5 What is Reranking?
Vector similarity can return "close but wrong" results. Reranking uses a secondary LLM to evaluate and re-order the search results by actual relevance to the question.

### 5.6 What is Query Rewriting?
In conversation, users often ask vague follow-ups ("Who won it?"). Query rewriting uses conversation history to convert these into specific, standalone queries that the retriever can handle.

---

## 6. Results & Comparison

| Metric | Brute-Force (Day 1) | Vector RAG (Day 3) | Advanced RAG (Day 5) |
|--------|---------------------|---------------------|-----------------------|
| **Search Method** | Keyword match | Cosine similarity | Similarity + reranking |
| **Handles Synonyms** | ❌ No | ✅ Yes | ✅ Yes |
| **Handles Follow-ups** | ❌ No | ❌ No | ✅ Yes (query rewriting) |
| **Chunk Quality** | N/A (full doc) | Mechanical splitting | LLM-generated headlines + summaries |
| **Result Accuracy** | Low | Good | Best |
| **API Cost** | Low | Medium | Higher (multiple LLM calls) |

---

## 7. Skills Demonstrated

- **RAG System Design** — from prototype to production-grade architecture
- **Vector Databases** — ChromaDB setup, embedding storage, similarity search
- **Text Embeddings** — local HuggingFace models and OpenAI embeddings
- **LangChain Framework** — document loading, text splitting, retrieval chains
- **Prompt Engineering** — context injection, system prompt design, temperature tuning
- **AI Evaluation** — measuring accuracy, comparing retrieval strategies
- **Advanced NLP** — semantic preprocessing, query rewriting, reranking
- **Structured Outputs** — Pydantic schemas for enforcing LLM response formats
- **UI Development** — Gradio chat interface for end-user interaction
- **Local AI** — Ollama integration for fully offline, cost-free operation
