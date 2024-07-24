import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
import sqlite3
import uuid
from typing import List, Dict, Any
from contextlib import contextmanager

# Initialize database


@contextmanager
def get_db_connection():
    conn = sqlite3.connect('chat_history.db')
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS chats
                     (id TEXT PRIMARY KEY, name TEXT, history TEXT)''')
        conn.commit()


@st.cache_resource
def load_vector_store():
    embeddings = OpenAIEmbeddings()
    loader = WebBaseLoader("https://docs.smith.langchain.com")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:50])
    return FAISS.from_documents(final_documents, embeddings)


def init_session_state():
    if "vector" not in st.session_state:
        st.session_state.vectors = load_vector_store()
        st.session_state.chat_history = []
        st.session_state.current_chat_id = None


def load_chat(chat_id: str = None):
    with get_db_connection() as conn:
        c = conn.cursor()
        if chat_id:
            c.execute("SELECT history FROM chats WHERE id=?", (chat_id,))
            result = c.fetchone()
            if result:
                st.session_state.chat_history = eval(result[0])
                st.session_state.current_chat_id = chat_id
            else:
                st.session_state.chat_history = []
                st.session_state.current_chat_id = None
        else:
            c.execute("SELECT id, history FROM chats ORDER BY rowid DESC LIMIT 1")
            result = c.fetchone()
            if result:
                st.session_state.chat_history = eval(result[1])
                st.session_state.current_chat_id = result[0]
            else:
                st.session_state.chat_history = []
                st.session_state.current_chat_id = None


def save_chat(chat_id: str, chat_name: str, history: List[Dict[str, Any]]):
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO chats (id, name, history) VALUES (?, ?, ?)",
                  (chat_id, chat_name, str(history)))
        conn.commit()


def get_chats():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id, name FROM chats ORDER BY rowid DESC")
        return c.fetchall()


def create_llm():
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")


def setup_retrieval_chain(llm):
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        </context>
        Question: {input}
        """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    return create_retrieval_chain(retriever, document_chain)


def main():
    init_db()
    init_session_state()

    st.title("ChatGROQ")

    # Sidebar for chat selection
    st.sidebar.title("Chat History")
    chats = get_chats()
    for chat in chats:
        if st.sidebar.button(chat[1], key=chat[0]):
            load_chat(chat[0])

    if st.sidebar.button("New Chat"):
        st.session_state.chat_history = []
        st.session_state.current_chat_id = None

    # Refresh button
    if st.sidebar.button("Refresh"):
        st.experimental_rerun()

    # Load the last chat if no specific chat is selected
    if not st.session_state.current_chat_id:
        load_chat()

    # Main chat interface
    chat_container = st.container()

    with chat_container:
        for chat in st.session_state.chat_history:
            st.write(f"**You:** {chat['input']}")
            st.write(f"**ChatGROQ:** {chat['response']}")
            if 'inference_time' in chat:
                st.write(
                    f"*Inference time: {chat['inference_time']:.4f} seconds*")
                st.write(
                    f"*Inference time per token: {chat['inference_time_per_token']:.6f} seconds*")
            st.write("---")
    prompt_input = st.text_input("Talk to me!", key="prompt_input")

    if prompt_input:
        llm = create_llm()
        retrieval_chain = setup_retrieval_chain(llm)

        start = time.time()
        response = retrieval_chain.invoke({"input": prompt_input})
        end = time.time()
        inference_time = end - start

        tokens = len(response['answer'].split())
        inference_time_per_token = inference_time / tokens if tokens > 0 else 0

        if st.session_state.current_chat_id:
            st.session_state.chat_history.append({
                "input": prompt_input,
                "response": response['answer'],
                "inference_time": inference_time,
                "inference_time_per_token": inference_time_per_token
            })

            save_chat(st.session_state.current_chat_id,
                      st.session_state.current_chat_id, st.session_state.chat_history)
        else:
            chat_id = str(uuid.uuid4())
            chat_name = prompt_input[:50]
            st.session_state.current_chat_id = chat_id
            st.session_state.chat_history = [{
                "input": prompt_input,
                "response": response['answer'],
                "inference_time": inference_time,
                "inference_time_per_token": inference_time_per_token
            }]

            save_chat(chat_id, chat_name, st.session_state.chat_history)

    # CSS to move input to bottom right and make it smaller
    st.markdown(
        """
        <style>
        .stTextInput {
            position: fixed;
            bottom: 20px;
            right: 200px;
            width: 60%;
        }
        .stTextInput > div > div > input {
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
