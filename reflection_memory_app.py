import streamlit as st
import asyncio
from reflection_memory import prepare_knowledge_base, rag_with_reflection, log_to_reflection_memory, embedding_function

st.set_page_config(page_title="RAG with Reflection Memory", layout="wide")
st.title("📄 RAG Chatbot with Reflection Memory")

# Session state for turns and chat history
if "turn" not in st.session_state:
    st.session_state.turn = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "knowledge_prepared" not in st.session_state:
    st.session_state.knowledge_prepared = False

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# Once uploaded, prepare the knowledge base
if uploaded_file and not st.session_state.knowledge_prepared:
    with st.spinner("Preparing knowledge base..."):
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        asyncio.run(prepare_knowledge_base("temp_uploaded.pdf"))
        st.session_state.knowledge_prepared = True
        st.success("Knowledge base ready!")

# Chat input
if st.session_state.knowledge_prepared:
    query = st.text_input("Ask a question:")
    if query:
        state = {"question": query}
        state = rag_with_reflection(state, st.session_state.turn)
        answer = state["answer"]

        # Display chat
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("RAG", answer))

        for role, msg in st.session_state.chat_history:
            if role == "You":
                st.markdown(f"**💬 You:** {msg}")
            else:
                st.markdown(f"**🤖 RAG:** {msg}")

        # Log to reflection memory
        log_to_reflection_memory(query, answer, state["context"], embedding_function)

        # Increment turn
        st.session_state.turn += 1