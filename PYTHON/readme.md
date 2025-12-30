````markdown
# LLAMA 3 Vector RAG (FAISS)  

A **Retrieval-Augmented Generation (RAG)** Python tool that integrates **LLAMA 3** with **FAISS** to answer questions based on multiple text files. It allows fast, context-aware responses by embedding text chunks and retrieving the most relevant ones for any user query.  

---

## Features

- ✅ Supports multiple text files as knowledge base  
- ✅ Embeds text using **Ollama embeddings** (`nomic-embed-text`)  
- ✅ Stores vectors in **FAISS** for fast similarity search  
- ✅ Updates the vector database automatically if source files change  
- ✅ Provides a **terminal-based interactive chat** with LLAMA 3  
- ✅ ANSI-colored outputs for better readability  

---

## Requirements

- Python 3.10+  
- Ollama Python SDK (`pip install ollama`)  
- FAISS (`pip install faiss-cpu` or `faiss-gpu`)  
- Numpy (`pip install numpy`)  

---

## Installation

1. Clone the repository or download the code.
2. Install dependencies:

```bash
pip install ollama faiss-cpu numpy
````

3. Make sure you have your **text files** (`data.txt`, `data2.txt`, etc.) in the same folder.
4. Run the script:

```bash
python3 your_script_name.py
```

---

## Usage

1. On first run, the script builds a **FAISS vector database** from your text files.
2. Once built, the database is reused unless the files are updated.
3. Type your query in the terminal. The AI will respond using **RAG context + general knowledge**.
4. Commands:

```text
exit or quit -> Stop the program
```

---

## Configuration

| Variable         | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `FILES`          | Array of text files to use as knowledge base         |
| `VECTOR_DB_FILE` | FAISS index file                                     |
| `META_FILE`      | Metadata file storing file hash and chunks           |
| `EMBED_MODEL`    | Embedding model (default: `nomic-embed-text`)        |
| `CHAT_MODEL`     | LLM model (default: `llama3`)                        |
| `CHUNK_SIZE`     | Number of characters per text chunk for embedding    |
| `TOP_K`          | Number of top similar chunks to retrieve for a query |

---

## How It Works

1. **Load files** → Split text into manageable chunks
2. **Compute embeddings** using Ollama
3. **Store embeddings** in FAISS for quick retrieval
4. **Retrieve top-K chunks** for each user query
5. **Send prompt** with context to LLAMA 3 and stream the answer

---

## Terminal UI

The program features a colorful ASCII terminal interface:

```
╔══════════════════════════════════════════════╗
║     🤖  LLAMA 3 VECTOR RAG (FAISS) 🤖        ║
║   Cold Start Once • Fast Forever             ║
╚══════════════════════════════════════════════╝
```

---

## Notes

* Ensure **Ollama API is installed and working**.
* The script is **offline-ready** once embeddings are generated.
* Works best with **plain text** knowledge files.

---


