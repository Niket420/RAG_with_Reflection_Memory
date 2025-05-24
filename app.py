import streamlit as st
import tempfile
import os
import asyncio
from main import read_pdf, splitting_pdfs, get_embeddings, retrieve, generate,self_reflection

st.set_page_config(page_title="RAG with Reflection Memory (Preview)", layout="wide")
st.title("📄 RAG Assistant with PDF Upload")

# Upload PDF/Doc
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    with st.spinner("🔍 Reading and indexing the document..."):
        pages = asyncio.run(read_pdf(temp_path))
        all_splits = splitting_pdfs(pages)
        get_embeddings(all_splits)

    st.success("✅ Document indexed. Ask your questions below!")
    os.remove(temp_path)  # Clean up

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("💬 Ask a question based on your uploaded PDF:")

if query:
    state = {"question": query}
    state.update(retrieve(state))
    state.update(generate(state))

    st.session_state.chat_history.append((query, state["answer"]))

   


# Display conversation history
if st.session_state.chat_history:
    st.subheader("📚 Conversation History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**RAG Assistant:** {a}")
        st.markdown("---")