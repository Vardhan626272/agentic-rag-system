# Local Agentic RAG Pipeline 🧠

A fully private, 100% local Retrieval-Augmented Generation (RAG) agent built with Python and LangChain. 

This project ingests complex PDF documents, processes them into a local vector database, and uses a local Large Language Model (Llama 3) to answer user queries based strictly on the provided context, complete with anti-hallucination guardrails.

## 🏗️ Architecture

* **Orchestration:** LangChain

* **LLM:** Llama 3 (via Ollama) - *Running locally for zero-cost inference*

* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`)

* **Vector Store:** ChromaDB (Local Persistence)

* **Document Loading:** PyPDFLoader

## 🚀 Key Features

* **Zero-Cost Infrastructure:** Ripped out paid APIs in favor of a fully local stack.

* **Hallucination Guardrails:** Engineered strict system prompts to prevent the agent from guessing outside the provided document context.

* **Efficient Chunking:** Utilized `RecursiveCharacterTextSplitter` for optimal context retrieval.

## 💻 Quick Start

1. Install Ollama and pull the Llama 3 model: `ollama run llama3`

2. Clone this repository.

3. Install dependencies: `pip install -r requirements.txt`

4. Place any target PDF in the root directory and rename it to `knowledge.pdf`.

5. Build the vector database: `python ingest.py`

6. Wake up the agent and start chatting: `python agent.py`