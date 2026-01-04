# 🎬 Logline Doctor – Agentic AI + RAG System

Logline Doctor is a **multi-agent GenAI application** that critiques weak movie loglines and rewrites them into compelling, market-ready pitches using **Agentic AI (AutoGen)** and **Retrieval-Augmented Generation (RAG)**.

Unlike simple chatbot demos, this project demonstrates **structured multi-agent reasoning**, **tool calling**, and **knowledge-grounded generation**.

---

## 🧠 What This Project Does

1. Accepts a weak, one-line movie logline from the user
2. An **Analyst Agent** critiques the logline based on proven screenwriting principles using RAG
3. The critique focuses on:
   - Protagonist  
   - Goal  
   - Conflict  
   - Stakes
4. A **Creative Writer Agent** rewrites the logline using the critique
5. Outputs a **clear, compelling, and commercially viable logline**

---

## 🧱 Architecture Overview

User (Streamlit / CLI)
↓
UserProxyAgent (AutoGen)
↓
AnalystAgent ──▶ RAG Tool (ChromaDB + Embeddings)
↓
CreativeWriterAgent
↓
Final Rewritten Logline


---

## 🧠 Core Concepts Demonstrated

- Agentic AI using role-based LLM agents (AutoGen)
- Tool calling from agents to external functions
- Retrieval-Augmented Generation (RAG)
- Vector database usage with ChromaDB
- Local HuggingFace embeddings for semantic search
- Separation of ingestion and inference phases
- Controlled, multi-step LLM workflows
- Reduced hallucinations through grounded context

---

## 🛠️ Tech Stack

- **Python**
- **AutoGen** – Multi-agent orchestration  
- **Groq (Llama 3.1)** – LLM inference  
- **LangChain** – Prompt & RAG pipelines  
- **ChromaDB** – Vector database  
- **HuggingFace Sentence Transformers** – Embeddings  
- **Streamlit** – UI  

---

## 📁 Project Structure (Single Folder)

.
├── agents.py / app.py # AutoGen agent workflow & orchestration
├── ingest.py # One-time RAG data ingestion
├── streamlit_app.py # Streamlit UI
├── logline_principles.txt # Screenwriting knowledge base
├── chroma_db/ # Vector DB (auto-generated, gitignored)
├── requirements.txt
├── .env
├── .gitignore
└── README.md


---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone <repo-url>
cd logline-doctor


2️⃣ Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set environment variables

Create a .env file:

GROQ_API_KEY=your_api_key_here

📦 RAG Data Ingestion (Run Once)

This step creates the local vector database.

python ingest.py


This will:

Load logline_principles.txt

Chunk the data

Generate embeddings

Persist them in chroma_db/

▶️ Run the Application
🔹 CLI Test (Agent workflow)
python agents.py

🔹 Streamlit UI
streamlit run streamlit_app.py
