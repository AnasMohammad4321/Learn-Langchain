from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

# Streamlit page configuration
st.set_page_config(page_title="ChatBot", page_icon="🤖")

st.title("Chatbot Interface")

st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose the model to use:",
    options=["OpenAI"] + ['deepseek-coder-v2', 'llama2', 'llama3', 'mistral']
)

if model_choice == "OpenAI":
    llm = ChatOpenAI(model="gpt-3.5-turbo")
else:
    llm = Ollama(model=model_choice)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if 'history' not in st.session_state:
    st.session_state.history = []

input_text = st.text_input("Type your message:")

if st.button("Send"):
    if input_text:
        st.session_state.history.append(f"You: {input_text}")

        result = chain.invoke({
            'question': input_text
        })

        # Add assistant message to history
        st.session_state.history.append(f"Bot: {result}")

        input_text = ""

# Display chat history
if st.session_state.history:
    for message in st.session_state.history:
        if message.startswith("You:"):
            st.markdown(
                f"<div style='text-align: left;'>{message}</div>", unsafe_allow_html=True)
        elif message.startswith("Bot:"):
            st.markdown(
                f"<div style='text-align: left;'>{message}</div>", unsafe_allow_html=True)
