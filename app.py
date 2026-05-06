import streamlit as st
import pdfplumber
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from groq import Groq

# ----------------------------
# CONFIG (GROQ CLIENT)
# ----------------------------
@st.cache_resource
def load_groq():
    return Groq(api_key=st.secrets["GROK"])  # Streamlit secret name

client = load_groq()

# ----------------------------
# LOAD EMBEDDING MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ----------------------------
# PDF TEXT EXTRACTION
# ----------------------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ----------------------------
# CHUNKING
# ----------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ----------------------------
# FAISS INDEX
# ----------------------------
def create_index(chunks):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index

# ----------------------------
# RETRIEVAL
# ----------------------------
def retrieve(query, index, chunks, k=3):
    query_vec = model.encode([query]).astype("float32")
    _, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

# ----------------------------
# GROQ RESPONSE
# ----------------------------
def ask_groq(context, question):
    try:
        prompt = f"""
You are a friendly AI Tutor.

1. Explain in simple words
2. Give a real-world example
3. Ask a short question to test understanding

Context:
{context}

Question:
{question}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # ✅ FIXED MODEL
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("📚 RAG AI Tutor (PDF-Based)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading PDF..."):
        text = extract_text(uploaded_file)

    if not text.strip():
        st.error("No text found in PDF.")
        st.stop()

    with st.spinner("Processing document..."):
        chunks = chunk_text(text)
        index = create_index(chunks)

    st.success("PDF ready!")

    # Input
    query = st.text_input("Ask a question from your PDF:")

    # BUTTON
    ask_button = st.button("🚀 Ask Question")

    if ask_button:
        if query.strip() == "":
            st.warning("Please type a question first.")
        else:
            with st.spinner("Thinking..."):
                retrieved_chunks = retrieve(query, index, chunks)
                context = "\n".join(retrieved_chunks)
                answer = ask_groq(context, query)

            st.markdown("### 📖 Answer")
            st.write(answer)
