"""
This module sets up and runs an interviewer agent using ChromaDB for
vector memory and RAG (Retrieval-Augmented Generation) capabilities.
It integrates with LMStudio and Ollama model clients for
conversational AI and document indexing.

Key functionalities:
- Configures ChromaDB for document storage and retrieval.
- Loads and indexes documents into vector memory.
- Sets up an assistant agent and user proxy for interactive conversations.
- Supports termination conditions for controlled chat sessions.
"""
import sys
import os
import asyncio
import chromadb

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelInfo
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig
)
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Custom import
sys.path.append("/home/jun/Documents/Master_IA/Code/TFM_BA_GPT/lib")
from indexing_helpers import load_and_index_documents

# -- MODEL SET UP -- #

model_info = ModelInfo(
    function_calling=True,  # Indica si el modelo soporta llamadas a funciones
    vision=False,           # Indica si el modelo tiene capacidades de visión
    json_output=True,      # Indica si el modelo puede generar salida en JSON
    family="unknown",       # Specify the model family if known
    structured_output=False,
)

# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.

# Model client for LMStudio
model_client = OpenAIChatCompletionClient(
    model="dolphin-2.9.3-mistral-nemo-12b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="LMStudio",
    model_info=model_info,
)

# Model client for Ollama
ollama_model_client = OllamaChatCompletionClient(
    model="qwen2.5:14b",
    host="https://ollama01.decoupled.ai",
    model_info=model_info)

# --- CONFIG PATH --- #
CHROMA_DIR = "/home/jun/Documents/Master_IA/Code/TFM_BA_GPT/chromadb_storage"
DOCS_DIR = "/home/jun/Documents/Master_IA/Code/TFM_BA_GPT/data"
COLLECTION_NAME = "rag_collection"

# --- EMBEDDING FUNCTION (Local SentenceTransformer) --- #
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# --- CHROMADB SET UP --- #
# Step 2: Set up ChromaDB client with embedding function
chroma_client = chromadb.Client(Settings(
    persist_directory=CHROMA_DIR,
    anonymized_telemetry=False
))

# Step 3: Create or load the collection WITH embedding
collections = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=emb_fn
)

# --- RAG-ENABLED AUTO-GEN SETUP WITH CHROMADB --- #
vector_memory = ChromaDBVectorMemory(
    config=PersistentChromaDBVectorMemoryConfig(
        collection_name=COLLECTION_NAME,
        persistence_path=CHROMA_DIR,
        k=3,
        score_threshold=0.4,
    )
)


async def main():
    """
    Main function to set up and run the interviewer agent.

    - Clears the vector memory and loads documents into it.
    - Inspects the ChromaDB collection to verify document indexing.
    - Sets up a user proxy and assistant agent for interactive conversations.
    - Runs a round-robin group chat session with termination conditions.
    - Closes the vector memory and model client after the session.
    """
    # Load and index documents
    if not os.path.exists(CHROMA_DIR):
        await vector_memory.clear()
        print("Indexing documents... please wait.")
        await load_and_index_documents(DOCS_DIR, vector_memory)

    # Check if the documents are indexed
    # Adjust the path to your DB directory
    # client = chromadb.PersistentClient(path=CHROMA_DIR)
    # collection = client.get_collection(COLLECTION_NAME)
    # print(collection.peek())  # Optional, to see the first few documents

    # --- AGENT SETUP --- #

    # Set up the Rag-enabled AutoGen agent and the user proxy
    # The user proxy is a special agent that represents
    # the human user in the conversation.
    # It can be used to send messages to the assistant agent
    # and receive responses.
    # The assistant agent is the main agent that interacts with
    # the user and performs tasks.

    user_proxy = UserProxyAgent(
        name="user_proxy",
        description="A human user",
        input_func=input
    )

    assistant_agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        memory=[vector_memory],
        # if you have a custom system prompt
        system_message=(
            "You are a helpful assistant that use RAG to answer questions"
        )
    )

    # Set up the termination word for the conversation
    # The termination condition is a special condition that indicates when
    # the conversation should end.
    # In this case, the conversation will end when the user types "exit".
    termination = TextMentionTermination("exit")

    # Create a round-robin group chat team with both agents
    team = RoundRobinGroupChat(
        [assistant_agent, user_proxy],
        termination_condition=termination,
    )

    # Function to run the chat
    initial_question = "I have a question about clean code. Can you help me?"
    stream = team.run_stream(task=initial_question)
    await Console(stream)  # stream the conversation to console

    await vector_memory.close()
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())



# New function to ask a question and get a response
async def get_agent_response(question: str) -> str:
    user_proxy = UserProxyAgent(
        name="user_proxy",
        description="A human user",
        input_func=None  # No need for input in Streamlit
    )

    assistant_agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        memory=[vector_memory],
        system_message="You are a helpful assistant that uses RAG to answer questions"
    )

    termination = TextMentionTermination("exit")
    team = RoundRobinGroupChat(
        [assistant_agent, user_proxy],
        termination_condition=termination,
    )

    async def run_chat():
        stream = team.run_stream(task=question)
        response = ""
        async for msg in stream:
            if msg.get("content"):
                response += msg["content"]
        return response

    response = await run_chat()
    return response
