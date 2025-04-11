from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
#from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
import logging
from autogen_agentchat import EVENT_LOGGER_NAME, TRACE_LOGGER_NAME
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import ModelInfo


#Model definition for model that are not supported natively by Autogen
model_info = ModelInfo(
    model_name="dolphin-2.9.3-mistral-nemo-12b",
    function_calling=True,  # Indica si el modelo soporta llamadas a funciones
    vision=False,           # Indica si el modelo tiene capacidades de visión
    json_output=True,      # Indica si el modelo puede generar salida en formato JSON
    family="unknown",       # Specify the model family if known
    structured_output=False
) 

# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.

#LMStudio Model
model_client = OpenAIChatCompletionClient(
    model="dolphin-2.9.3-mistral-nemo-12b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="LMStudio", 
    model_info=model_info,
)

#Ollama Model
ollama_model_client = OllamaChatCompletionClient(
    model="qwen2.5:14b", 
    host="https://ollama01.decoupled.ai", 
    model_info=model_info
    )

### --- CONFIG Path--- ###
CHROMA_DIR = "../chromadb_storage"
DOCS_DIR = "../data"
COLLECTION_NAME = "rag_collection"

### --- EMBEDDING FUNCTION (Local SentenceTransformer) --- ###
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

#Function to load documents from the specified directory and index them
# using the ChromaDB vector store.
# This function loads documents from the DOCS_DIR, splits them into smaller chunks,
# and indexes them in the ChromaDB vector store.
# It uses the RecursiveCharacterTextSplitter to split the documents into chunks
# of a specified size with a specified overlap.
# The function returns the collection object from ChromaDB.
# The documents can be in PDF, TXT, or DOCX format.
def load_and_index_documents():
    docs = []
    for filename in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    texts = [doc.page_content for doc in split_docs]
    metadatas = [doc.metadata for doc in split_docs]

    client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=[str(i) for i in range(len(texts))]
    )
    return collection

### --- RAG-ENABLED AUTO-GEN SETUP --- ###
vector_memory = ChromaDBVectorMemory(
    config=PersistentChromaDBVectorMemoryConfig(
        collection_name=COLLECTION_NAME,
        persistence_path=CHROMA_DIR,
        k=3,
        score_threshold=0.4,
    )
)

#Run the load and index function
if not os.path.exists(CHROMA_DIR):
        print("Indexing documents...")
        load_and_index_documents()
        print("Documents indexed.")

# Create a group chat with the user proxy and assistant agent
user_proxy = UserProxyAgent(
    name="user_proxy",
    description="A human user",
    input_func=input
)


assistant_agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    memory=[vector_memory],
    #system_message="You are a helpful assistant"  # if you have a custom system prompt
)

termination = TextMentionTermination("exit")

# Create a round-robin group chat team with both agents
team = RoundRobinGroupChat(
    [assistant_agent, user_proxy], 
    termination_condition=termination
)



stream = assistant_agent.run_stream(task="what are the  books that all serious practitioners will have on their bookshelves?")
await Console(stream)

await vector_memory.close()
await model_client.close()