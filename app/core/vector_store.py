"""
Vector Database Integration Module

This module handles the embedding and storage of document chunks in a vector database
for semantic search and retrieval.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages the storage and retrieval of document chunks in a vector database.
    Uses ChromaDB as the underlying vector store with sentence-transformers for embeddings.
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "insurance_policies"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector database
            embedding_model: Name of the sentence-transformers model to use
            collection_name: Name of the collection in the vector database
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Create embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Using existing collection: {collection_name}")
        except ValueError:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_document(self, 
                    document_id: str,
                    chunks: List[Dict[str, Any]],
                    metadata: Dict[str, Any]) -> None:
        """
        Add document chunks to the vector database.
        
        Args:
            document_id: Unique identifier for the document
            chunks: List of document chunks with text and metadata
            metadata: Document-level metadata
        """
        logger.info(f"Adding document {document_id} with {len(chunks)} chunks to vector store")
        
        # Prepare data for batch addition
        ids = []
        texts = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Create a unique ID for each chunk
            chunk_id = f"{document_id}_chunk_{i}"
            
            # Get chunk text
            chunk_text = chunk["text"]
            
            # Combine document metadata with chunk metadata
            chunk_metadata = metadata.copy()
            chunk_metadata.update(chunk["metadata"])
            
            # Convert all metadata values to strings for ChromaDB compatibility
            for key, value in chunk_metadata.items():
                if not isinstance(value, (str, int, float, bool)):
                    chunk_metadata[key] = str(value)
            
            # Add to batch lists
            ids.append(chunk_id)
            texts.append(chunk_text)
            metadatas.append(chunk_metadata)
        
        # Add to collection in batch
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully added document {document_id} to vector store")
    
    def search(self, 
              query: str, 
              filter_criteria: Optional[Dict[str, Any]] = None,
              limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks based on a query.
        
        Args:
            query: Search query
            filter_criteria: Optional filter criteria for metadata
            limit: Maximum number of results to return
            
        Returns:
            List of relevant document chunks with text and metadata
        """
        logger.info(f"Searching for: {query}")
        
        # Execute search
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=filter_criteria
        )
        
        # Format results
        formatted_results = []
        
        if results["documents"]:
            documents = results["documents"][0]  # First query results
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]
            
            for i in range(len(documents)):
                formatted_results.append({
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i],
                    "id": ids[i]
                })
        
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of document chunks with text and metadata
        """
        # Query for chunks with the document ID prefix
        results = self.collection.query(
            query_texts=None,
            where={"document_id": document_id},
            n_results=1000  # Large number to get all chunks
        )
        
        # Format results
        formatted_results = []
        
        if results["documents"]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            ids = results["ids"][0]
            
            for i in range(len(documents)):
                formatted_results.append({
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "id": ids[i]
                })
        
        return formatted_results
    
    def delete_document(self, document_id: str) -> None:
        """
        Delete all chunks for a specific document.
        
        Args:
            document_id: Document identifier
        """
        logger.info(f"Deleting document {document_id} from vector store")
        
        # Delete chunks with the document ID prefix
        self.collection.delete(
            where={"document_id": document_id}
        )
    
    def get_similar_chunks(self, 
                          chunk_id: str, 
                          limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: ID of the chunk to find similar chunks for
            limit: Maximum number of results to return
            
        Returns:
            List of similar chunks with text and metadata
        """
        # Get the chunk
        result = self.collection.get(
            ids=[chunk_id],
            include=["embeddings", "documents", "metadatas"]
        )
        
        if not result["embeddings"]:
            return []
        
        # Get the embedding
        embedding = result["embeddings"][0]
        
        # Search for similar chunks
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit + 1  # +1 because the chunk itself will be included
        )
        
        # Format results, excluding the chunk itself
        formatted_results = []
        
        if results["documents"]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]
            
            for i in range(len(documents)):
                if ids[i] != chunk_id:  # Exclude the original chunk
                    formatted_results.append({
                        "text": documents[i],
                        "metadata": metadatas[i],
                        "distance": distances[i],
                        "id": ids[i]
                    })
        
        return formatted_results
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embedding_model.encode(text).tolist()
    
    def reset(self) -> None:
        """Reset the vector store by deleting and recreating the collection."""
        logger.warning(f"Resetting vector store collection: {self.collection_name}")
        
        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        
        logger.info(f"Vector store reset complete")
