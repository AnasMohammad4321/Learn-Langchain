import requests
import streamlit as st


def get_api_response(endpoint, input_text):
    try:
        response = requests.post(
            f"http://localhost:8000/{endpoint}/invoke",
            json={'input': {'topic': input_text}}
        )
        response.raise_for_status()
        data = response.json()
        return data.get('output', 'No content found')
    except Exception as e:
        return f"Error: {str(e)}"


st.set_page_config(page_title="Langchain Using APIs", layout="wide")

st.title('Langchain Using APIs')

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Model Selection")
    model_choice = st.radio(
        "Choose a model:",
        options=["OpenAI", 'deepseek-coder-v2', 'llama2', 'llama3', 'mistral']
    )

    if model_choice == "OpenAI":
        task = "Essay"
        endpoint = "essay"
    else:
        task = "Poem"
        endpoint = "poem"

    input_text = st.text_area(f"Enter the topic for the {task}:", height=100)

    if st.button(f"Generate {task}"):
        if input_text:
            with st.spinner(f"Generating {task}..."):
                response = get_api_response(endpoint, input_text)
            st.success(f"{task} generated successfully!")
        else:
            st.warning("Please enter a topic.")

with col2:
    st.subheader(f"{task} Output")
    if 'response' in locals():
        st.write(response)
    else:
        st.info(f"Your generated {task.lower()} will appear here.")
