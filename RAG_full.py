import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# ------------------------------
# Configuration
# ------------------------------
DATA_FOLDER = "rag_texts"
FAISS_INDEX_PATH = "faiss_index"

# ------------------------------
# 1️⃣ Load documents with metadata
# ------------------------------
docs = []
for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".txt"):
        path = os.path.join(DATA_FOLDER, filename)
        with open(path, "r", encoding="utf-8") as f:
            docs.append(
                Document(
                    page_content=f.read(),
                    metadata={"source": filename}
                )
            )

print(f"Total documents loaded: {len(docs)}")

# ------------------------------
# 2️⃣ Split documents into chunks (metadata preserved)
# ------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)
print(f"Total chunks created: {len(chunks)}")

# ------------------------------
# 3️⃣ Local embeddings (no OpenAI API)
# ------------------------------
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------
# 4️⃣ Create or load FAISS index
# ------------------------------
if os.path.exists(FAISS_INDEX_PATH):
    print("Loading existing FAISS index...")
    vs = FAISS.load_local(
        FAISS_INDEX_PATH,
        emb,
        allow_dangerous_deserialization=True
    )
else:
    print("Creating FAISS index locally (no quota)...")
    batch_size = 50
    vs = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Embedding batch {i} to {i + len(batch)}...")

        if vs is None:
            vs = FAISS.from_texts(batch, embedding=emb)
        else:
            vs.add_texts(batch)

    vs.save_local(FAISS_INDEX_PATH)
    print("FAISS index saved locally.")

retriever = vs.as_retriever(search_kwargs={"k": 5})

# ------------------------------
# 5️⃣ LLM for answer synthesis
# ------------------------------
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# ------------------------------
# 6️⃣ Query function with sources
# ------------------------------
def answer_query(query: str) -> str:
    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    # Build context with source citations
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )

    # Build prompt
    prompt = (
        "Use the context below to answer the question as accurately as possible.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    # Generate answer
    resp = llm.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)

# ------------------------------
# 7️⃣ Interactive loop
# ------------------------------
print("\nClinical Trial Intelligence Assistant ready! Type 'exit' to quit.\n")
while True:
    q = input("Ask a question about pancreatic cancer trials: ")
    if q.lower() in ["exit", "quit"]:
        break
    ans = answer_query(q)
    print("\nAnswer:\n", ans)
    print("-" * 80)
