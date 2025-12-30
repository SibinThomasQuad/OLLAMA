import ollama         # Ollama API to interact with LLaMA3 model and embeddings
import os
import sys
import hashlib        # For computing file hashes to check if data changed
import faiss          # Vector database for fast similarity search
import pickle         # To save/load metadata
import numpy as np    # For handling vectors efficiently

# ===== ANSI COLORS FOR TERMINAL =====
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"

# ===== CONFIGURATION =====
FILES = ["data.txt", "data2.txt", "data3.txt"]  # Array of text files to use
VECTOR_DB_FILE = "vector.index"  # FAISS index file
META_FILE = "meta.pkl"           # Metadata (hash, chunks)
EMBED_MODEL = "nomic-embed-text" # Embedding model
CHAT_MODEL = "llama3"            # LLM model
CHUNK_SIZE = 1000                # Size of text chunks to embed
TOP_K = 5                        # Number of top similar chunks to retrieve

# ===== GLOBAL SYSTEM PROMPT =====
SYSTEM_PROMPT = """You are a server support assistant.
Use the context below to answer the user's question.
If the context is not sufficient, provide general guidance based on your knowledge."""

# ===== UTILITY FUNCTIONS =====
def files_hash(file_list):
    """Compute MD5 hash of all files combined."""
    md5 = hashlib.md5()
    for file in file_list:
        if os.path.exists(file):
            with open(file, "rb") as f:
                md5.update(f.read())
    return md5.hexdigest()

def load_chunks_from_files(file_list):
    """Load and combine text from multiple files, split into chunks."""
    chunks, buf = [], ""
    for file in file_list:
        if not os.path.exists(file):
            continue
        with open(file, "r", encoding="utf-8") as f:
            text = f.read().lower()  # lowercase for embedding consistency
        for line in text.splitlines():
            if len(buf) + len(line) < CHUNK_SIZE:
                buf += line + "\n"
            else:
                chunks.append(buf.strip())
                buf = line + "\n"
    if buf.strip():
        chunks.append(buf.strip())
    return chunks

# ===== BUILD OR LOAD FAISS VECTOR DATABASE =====
def build_or_load_db():
    """Load existing FAISS DB if available, otherwise build it."""
    current_hash = files_hash(FILES)
    if os.path.exists(VECTOR_DB_FILE) and os.path.exists(META_FILE):
        with open(META_FILE, "rb") as f:
            meta = pickle.load(f)
        if meta.get("hash") == current_hash:
            print(f"{GREEN}✅ Loaded FAISS vector DB{RESET}\n")
            index = faiss.read_index(VECTOR_DB_FILE)
            return index, meta["chunks"]

    print(f"{YELLOW}🔄 Building FAISS DB embeddings (one-time)...{RESET}")
    chunks = load_chunks_from_files(FILES)
    embeddings = []

    for i, chunk in enumerate(chunks, 1):
        emb = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)["embedding"]
        embeddings.append(emb)
        print(f"{CYAN}✔ Chunk {i}/{len(chunks)} embedded{RESET}")

    vectors = np.array(embeddings).astype("float32")
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)   # Normalize for cosine similarity
    index.add(vectors)             # Add vectors to index

    faiss.write_index(index, VECTOR_DB_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"hash": current_hash, "chunks": chunks}, f)

    print(f"{GREEN}✅ FAISS DB built and saved{RESET}\n")
    return index, chunks

# ===== RETRIEVE TOP-K CHUNKS FOR A QUERY =====
def retrieve_context(question, index, chunks):
    """Return the top-K most similar chunks for a given question."""
    q_vec = ollama.embeddings(model=EMBED_MODEL, prompt=question.lower())["embedding"]
    q_vec = np.array([q_vec]).astype("float32")
    faiss.normalize_L2(q_vec)

    D, I = index.search(q_vec, TOP_K)
    context = "\n---\n".join(chunks[i] for i in I[0])
    return context

# ===== TERMINAL UI =====
def show_title():
    """Show a nice ASCII header in terminal."""
    print(f"""{MAGENTA}{BOLD}
╔══════════════════════════════════════════════╗
║     🤖  LLAMA 3 VECTOR RAG (FAISS) 🤖        ║
║   Cold Start Once • Fast Forever             ║
╚══════════════════════════════════════════════╝
{RESET}""")

# ===== MAIN LOOP =====
show_title()
index, chunks = build_or_load_db()

print(f"{YELLOW}Type 'exit' or 'quit' to stop.{RESET}\n")

while True:
    try:
        q = input(f"{GREEN}You:{RESET} ").strip()
        if q.lower() in ("exit", "quit"):
            print(f"\n{RED}👋 Goodbye!{RESET}")
            break

        # Retrieve context from vector DB
        context = retrieve_context(q, index, chunks)

        # Prepare prompt using global SYSTEM_PROMPT
        prompt = f"""{SYSTEM_PROMPT}

Context:
---------
{context}
---------

Question:
{q}
"""

        # Stream response from LLaMA3
        stream = ollama.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        print(f"{CYAN}AI:{RESET} ", end="", flush=True)
        for c in stream:
            if "message" in c and "content" in c["message"]:
                sys.stdout.write(c["message"]["content"])
                sys.stdout.flush()
        print("\n")

    except KeyboardInterrupt:
        print(f"\n{RED}Interrupted{RESET}")
        break
