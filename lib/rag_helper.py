"""
This module provides helper functions for creating and managing ChromaDB-based
vector memory for retrieval-augmented generation (RAG) workflows.

The primary function, `create_chromadb_memory`, initializes a ChromaDB
collection with a local SentenceTransformer embedding function and
returns a vector memory object for use in RAG-enabled applications.
"""
import chromadb
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig
)
from chromadb.config import Settings
from chromadb.utils import embedding_functions


async def create_chromadb_memory(
    chroma_dir: str, collection_name: str
) -> ChromaDBVectorMemory:
    """
    Create a ChromaDB collection with a local SentenceTransformer embedding
    function.

    Args:
        CHROMA_DIR (str): Directory for ChromaDB persistence.
        DATA_DIR (str): Directory containing the documents to be indexed.
        # The documents should be in PDF, TXT, or DOCX format.
        COLLECTION_NAME (str): Name of the collection to create or load.

    Returns:
        chroma_client: ChromaDB client with the specified collection.
    """

    # --- EMBEDDING FUNCTION (Local SentenceTransformer) --- #
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # --- CHROMADB SET UP --- #
    # Step 2: Set up ChromaDB client with embedding function
    chroma_client = chromadb.Client(Settings(
        persist_directory=chroma_dir,
        anonymized_telemetry=False
    ))

    # Step 3: Create or load the collection with embedding
    chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn
        )

    # --- RAG-ENABLED AUTO-GEN SETUP WITH CHROMADB --- #
    vector_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name=collection_name,
            persistence_path=chroma_dir,
            k=3,
            score_threshold=0.4,
        )
    )
    return vector_memory
