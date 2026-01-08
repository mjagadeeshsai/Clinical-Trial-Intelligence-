import streamlit as st
from RAG_full import answer_query  # your RAG logic
from dotenv import load_dotenv
import os

# Load .env for API key
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Clinical Trial RAG Assistant",
    page_icon="🧬",
    layout="wide"
)

# --- Custom CSS for background and styling ---
st.markdown("""
    <style>
    /* Background color */
    .stApp {
        background: #f0f4f8;
        color: #111;
    }

    /* Sidebar styling */
    .css-1d391kg {  /* sidebar container class may vary by Streamlit version */
        background-color: #e0e7ff;
        padding: 20px;
        border-radius: 10px;
    }

    /* Title styling */
    .stTitle {
        color: #1a1a2e;
    }

    /* Input box styling */
    div.stTextInput>div>div>input {
        background-color: #ffffff;
        border: 2px solid #b0b7ff;
        border-radius: 8px;
        padding: 10px;
        color: #000;
    }

    /* Success box */
    .stAlert {
        border-radius: 10px;
        padding: 15px;
    }

    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🧬 Clinical Trial RAG Assistant")
st.sidebar.markdown("""
Ask questions about pancreatic cancer clinical trials and get structured, source-aware answers.
- Embeddings are computed locally using HuggingFace Transformers
- GPT-4.1-mini generates human-readable answers from retrieved documents
""")

# --- Main Page ---
st.title("🧬 Clinical Trial RAG Assistant")
st.markdown("**Ask questions about pancreatic cancer trials and get structured, context-aware answers.**")

# --- Input ---
query = st.text_input("Enter your question here:")

# --- Button / Response ---
if query:
    with st.spinner("Fetching answer..."):
        try:
            answer = answer_query(query)
            st.success("✅ Answer retrieved!")
            st.markdown(f"**Question:** {query}")
            st.markdown(f"**Answer:** {answer}")
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")

# --- Footer ---
st.markdown("""
---
Made with ❤️ using Streamlit & LangChain
""")
