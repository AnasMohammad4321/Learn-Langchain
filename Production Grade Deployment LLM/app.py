from fastapi import FastAPI
from langchain_openai import ChatOpenAI  # Updated import
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Create FastAPI instance
app = FastAPI(
    title="Langchain Server",
    version="1.0",  # Corrected version key
    description="A simple API Server"
)

# Initialize models
openai_model = ChatOpenAI()
llama_model = Ollama(model="llama2")

# Define prompts
prompt1 = ChatPromptTemplate.from_template(
    "Write me an essay about {topic} around 100 words"
)

prompt2 = ChatPromptTemplate.from_template(
    "Write me a poem about {topic} for a 5-year-old child with 100 words"
)

# Add routes
add_routes(
    app,
    openai_model,
    path="/openai"
)

add_routes(
    app,
    prompt1 | openai_model,
    path="/essay"
)

add_routes(
    app,
    prompt2 | llama_model,
    path="/poem"
)

# Run the application
if __name__ == "__main__":  # Corrected name check
    uvicorn.run(app, host="localhost", port=8000)
