# Clinical Trial RAG Assistant

A **Retrieval-Augmented Generation (RAG) system** for answering questions about pancreatic cancer clinical trials.  
The system combines **local embeddings** for document retrieval with **GPT-4.1-mini** for high-level answer generation.

> ⚡ Key point: All document embeddings are computed locally using HuggingFace transformers, so no API calls are needed for retrieval. GPT-4 is only used for generating human-readable answers from the retrieved context.

---

## Features

- Embed clinical trial documents locally using **sentence-transformers/all-MiniLM-L6-v2**
- Use **FAISS vector store** for fast and scalable document retrieval
- Generate structured, context-aware answers with **GPT-4.1-mini**
- Optional source-aware answers (document filenames displayed)
- Handles complex queries like:
  - Number of vaccine-related trials
  - Average number of patients per trial
  - Phase-specific dosing schedules
  - Trial cohort comparisons

---

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt

Create a file called .env in your project root (same level as src/) and add your key like this:

OPENAI_API_KEY=your_api_key_here


⚠️ Note: This is only required for GPT-4 answer generation. The embeddings are local, so no API is needed for retrieval.

**2️. Add your clinical trial documents**

Create a folder named rag_texts/ in your project root (same level as src/).

Put all your sample clinical trial .txt files into this folder.

Each .txt file should contain the content of a single clinical trial (you can have multiple files).

**3️. Run the assistant**

Open a terminal in your project folder and run:

python src/RAG_full.py


The script will load your documents, create (or load) the FAISS embeddings locally, and then start an interactive session.

**4️. Ask questions interactively**

Once it’s running, you’ll see a prompt:

Ask a question about pancreatic cancer trials:


**4.1 Examples of questions you can ask:**

How many vaccine-related trials are there?
What is the average number of patients per trial?
List the phases and dosing schedules of trials.
Compare cohort treatments for GVAX and Ras peptide vaccines.


**5. Type exit or quit to end the session.**
------------------

**Tech Stack**
Python 3.10+
LangChain
FAISS
HuggingFace Transformers (for local embeddings)
OpenAI GPT-4.1-mini (for answer generation only)

**Notes / Tips**
1. FAISS index is cached locally for faster future runs
2. Batch size for embeddings can be adjusted if memory issues occur
3. GPT-4 is only used for generating readable answers, all retrieval is local
4. Works entirely offline for embeddings; internet needed only for GPT-4 calls


