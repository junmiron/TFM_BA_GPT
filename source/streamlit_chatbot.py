import streamlit as st
import asyncio
from interviewer_agent import get_agent_response

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("💬 RAG-powered Chatbot")
st.markdown("Ask a question about your documents and get intelligent answers using RAG.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box
if prompt := st.chat_input("Ask something..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(get_agent_response(prompt))
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
