"""
This module provides utilities for loading and indexing documents
into vector memory. It supports PDF, TXT, and DOCX formats and
uses a text splitter to divide documents into smaller chunks
for efficient indexing.
"""

import os
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from autogen_core.memory import MemoryContent, MemoryMimeType

# --- INDEXING UTILITIES --- #

# Funtion to load and index documents
# This function loads documents from a specified directory,
# splits them into chunks, and indexes them in the provided vector memory.
# It supports PDF, TXT, and DOCX formats.
# The function uses a text splitter to divide the documents into smaller chunks
# based on the specified chunk size and overlap.
# It also handles exceptions for unsupported file formats and
# errors during processing.


async def load_and_index_documents(
    docs_dir,
    vector_memory,
    chunk_size=500,
    chunk_overlap=50
):
    """
    Loads PDF, TXT, and DOCX documents from the specified directory,
    splits them into chunks, and indexes them in the provided
    AutoGen vector memory.

    Args:
        docs_dir (str): Path to the directory containing the documents.
        vector_memory (ChromaDBVectorMemory): The vector memory object to
        index the documents into.
        chunk_size (int): Number of characters in each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for file_name in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, file_name)
        ext = os.path.splitext(file_name)[-1].lower()

        try:
            if ext == ".pdf":
                loader = PyMuPDFLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path)
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                print(f"Unsupported file format: {file_name}")
                continue

            docs = loader.load()
            split_docs = text_splitter.split_documents(docs)

            for i, chunk in enumerate(split_docs):
                await vector_memory.add(  # Add `await` here
                    MemoryContent(
                        content=chunk.page_content,
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"source": file_name, "chunk_index": i}
                    )
                )

            print(f"Indexed {len(split_docs)} chunks from {file_name}.")

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"Error processing {file_name}: {e}")
    return chunk_size, chunk_overlap
