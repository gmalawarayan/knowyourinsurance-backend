"""
LLM Query System Module

This module handles the interaction with language models for answering queries
about insurance policy documents using retrieval-augmented generation.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
import json
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

from app.core.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMQuerySystem:
    """
    Handles natural language queries about insurance policies using LLMs and RAG.
    """
    
    def __init__(self, 
                 vector_store: VectorStore,
                 model_type: str = "ollama",
                 model_name: str = "llama3",
                 api_key: Optional[str] = None,
                 temperature: float = 0.1):
        """
        Initialize the LLM query system.
        
        Args:
            vector_store: Vector store for document retrieval
            model_type: Type of model to use ('ollama' or 'openai')
            model_name: Name of the model to use
            api_key: API key for OpenAI (if using OpenAI)
            temperature: Temperature for model generation
        """
        self.vector_store = vector_store
        self.model_type = model_type
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LLM based on model type
        if model_type == "ollama":
            logger.info(f"Initializing Ollama LLM with model: {model_name}")
            self.llm = Ollama(model=model_name, temperature=temperature)
        elif model_type == "openai":
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key is required for OpenAI models")
            
            logger.info(f"Initializing OpenAI LLM with model: {model_name}")
            self.llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Initialize prompt templates
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize prompt templates for different query types."""
        
        # General query prompt
        self.general_query_prompt = PromptTemplate.from_template(
            """You are an AI assistant specialized in analyzing insurance policies.
            
            Answer the question based only on the following context:
            
            {context}
            
            Question: {question}
            
            Provide a detailed and accurate answer. If the information is not in the context, 
            say "I don't have enough information to answer this question based on the provided policy."
            
            Answer:"""
        )
        
        # Coverage analysis prompt
        self.coverage_analysis_prompt = PromptTemplate.from_template(
            """You are an AI assistant specialized in analyzing insurance policies.
            
            Analyze the coverage details in the following insurance policy context:
            
            {context}
            
            Question about coverage: {question}
            
            Provide a detailed analysis of the coverage. Include specific coverage limits, 
            conditions, and any potential gaps or limitations. If the information is not in the context, 
            say "I don't have enough information to analyze this coverage based on the provided policy."
            
            Coverage Analysis:"""
        )
        
        # Exclusion analysis prompt
        self.exclusion_analysis_prompt = PromptTemplate.from_template(
            """You are an AI assistant specialized in analyzing insurance policies.
            
            Analyze the exclusions in the following insurance policy context:
            
            {context}
            
            Question about exclusions: {question}
            
            Provide a detailed analysis of what is NOT covered by this policy. Explain the exclusions 
            in plain language and their implications. If the information is not in the context, 
            say "I don't have enough information to analyze the exclusions based on the provided policy."
            
            Exclusion Analysis:"""
        )
        
        # Premium analysis prompt
        self.premium_analysis_prompt = PromptTemplate.from_template(
            """You are an AI assistant specialized in analyzing insurance policies.
            
            Analyze the premium details in the following insurance policy context:
            
            {context}
            
            Question about premium: {question}
            
            Provide a detailed analysis of the premium structure, including total cost, 
            payment schedule, and any factors affecting the premium. If the information is not in the context, 
            say "I don't have enough information to analyze the premium based on the provided policy."
            
            Premium Analysis:"""
        )
    
    def query(self, 
             question: str, 
             document_id: Optional[str] = None,
             query_type: str = "general",
             k: int = 5) -> Dict[str, Any]:
        """
        Answer a question about an insurance policy using RAG.
        
        Args:
            question: User's question
            document_id: Optional document ID to restrict search to
            query_type: Type of query ('general', 'coverage', 'exclusion', 'premium')
            k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and source chunks
        """
        logger.info(f"Processing query: {question}")
        
        # Determine filter criteria
        filter_criteria = None
        if document_id:
            filter_criteria = {"document_id": document_id}
        
        # Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(
            query=question,
            filter_criteria=filter_criteria,
            limit=k
        )
        
        if not retrieved_chunks:
            return {
                "answer": "I couldn't find any relevant information in the policy to answer your question.",
                "sources": []
            }
        
        # Format context from chunks
        context = self._format_context(retrieved_chunks)
        
        # Select appropriate prompt based on query type
        if query_type == "coverage":
            prompt = self.coverage_analysis_prompt
        elif query_type == "exclusion":
            prompt = self.exclusion_analysis_prompt
        elif query_type == "premium":
            prompt = self.premium_analysis_prompt
        else:  # default to general
            prompt = self.general_query_prompt
        
        # Create and run chain
        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Execute chain
        answer = chain.invoke(question)
        
        # Format sources
        sources = self._format_sources(retrieved_chunks)
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def analyze_coverage_gaps(self, document_id: str) -> Dict[str, Any]:
        """
        Analyze coverage gaps in an insurance policy.
        
        Args:
            document_id: Document ID of the policy to analyze
            
        Returns:
            Dictionary with coverage gap analysis
        """
        logger.info(f"Analyzing coverage gaps for document: {document_id}")
        
        # Retrieve policy chunks
        chunks = self.vector_store.get_document_chunks(document_id)
        
        if not chunks:
            return {
                "analysis": "No policy document found with the provided ID.",
                "gaps": []
            }
        
        # Format context from chunks
        context = self._format_context(chunks)
        
        # Create coverage gap analysis prompt
        coverage_gap_prompt = PromptTemplate.from_template(
            """You are an AI assistant specialized in analyzing insurance policies.
            
            Analyze the following insurance policy for potential coverage gaps:
            
            {context}
            
            Identify at least 3-5 potential coverage gaps or limitations in this policy.
            For each gap, provide:
            1. A brief description of the gap
            2. The potential risk it poses to the policyholder
            3. A recommendation to address this gap
            
            Format your response as a JSON object with the following structure:
            {{
                "summary": "Brief overall assessment of the policy's coverage",
                "gaps": [
                    {{
                        "description": "Description of the gap",
                        "risk": "Potential risk to policyholder",
                        "recommendation": "Recommendation to address the gap"
                    }},
                    ...
                ]
            }}
            
            Coverage Gap Analysis:"""
        )
        
        # Create and run chain
        chain = (
            {"context": lambda _: context}
            | coverage_gap_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Execute chain
        result = chain.invoke("")
        
        # Parse JSON result
        try:
            analysis = json.loads(result)
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw text
            analysis = {
                "summary": "Analysis could not be properly formatted.",
                "gaps": [],
                "raw_analysis": result
            }
        
        return analysis
    
    def compare_policies(self, document_id_1: str, document_id_2: str) -> Dict[str, Any]:
        """
        Compare two insurance policies.
        
        Args:
            document_id_1: Document ID of the first policy
            document_id_2: Document ID of the second policy
            
        Returns:
            Dictionary with policy comparison
        """
        logger.info(f"Comparing policies: {document_id_1} and {document_id_2}")
        
        # Retrieve policy chunks
        chunks_1 = self.vector_store.get_document_chunks(document_id_1)
        chunks_2 = self.vector_store.get_document_chunks(document_id_2)
        
        if not chunks_1 or not chunks_2:
            return {
                "comparison": "One or both policy documents could not be found.",
                "differences": []
            }
        
        # Format context from chunks
        context_1 = self._format_context(chunks_1)
        context_2 = self._format_context(chunks_2)
        
        # Create policy comparison prompt
        comparison_prompt = PromptTemplate.from_template(
            """You are an AI assistant specialized in analyzing insurance policies.
            
            Compare the following two insurance policies:
            
            POLICY 1:
            {context_1}
            
            POLICY 2:
            {context_2}
            
            Provide a detailed comparison of these policies, focusing on:
            1. Coverage differences
            2. Premium differences
            3. Deductible differences
            4. Exclusion differences
            5. Overall value comparison
            
            Format your response as a JSON object with the following structure:
            {{
                "summary": "Brief overall comparison of the two policies",
                "differences": [
                    {{
                        "category": "Category name (e.g., Coverage, Premium)",
                        "policy1": "Details from policy 1",
                        "policy2": "Details from policy 2",
                        "assessment": "Which policy is better in this category and why"
                    }},
                    ...
                ],
                "recommendation": "Overall recommendation on which policy provides better value"
            }}
            
            Policy Comparison:"""
        )
        
        # Create and run chain
        chain = (
            {"context_1": lambda _: context_1, "context_2": lambda _: context_2}
            | comparison_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Execute chain
        result = chain.invoke("")
        
        # Parse JSON result
        try:
            comparison = json.loads(result)
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw text
            comparison = {
                "summary": "Comparison could not be properly formatted.",
                "differences": [],
                "raw_comparison": result
            }
        
        return comparison
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a context string.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            # Format chunk with metadata
            section = chunk.get("metadata", {}).get("section", "Unknown Section")
            context_parts.append(f"--- CHUNK {i+1} (SECTION: {section}) ---\n{chunk['text']}\n")
        
        return "\n".join(context_parts)
    
    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format source chunks for response.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            List of formatted source dictionaries
        """
        sources = []
        
        for chunk in chunks:
            source = {
                "text": chunk["text"],
                "section": chunk.get("metadata", {}).get("section", "Unknown Section"),
                "document_id": chunk.get("metadata", {}).get("document_id", "Unknown Document"),
                "relevance_score": 1.0 - chunk.get("distance", 0) if "distance" in chunk else 1.0
            }
            sources.append(source)
        
        return sources
    
    def detect_query_type(self, question: str) -> str:
        """
        Detect the type of query based on the question.
        
        Args:
            question: User's question
            
        Returns:
            Query type ('general', 'coverage', 'exclusion', 'premium')
        """
        question_lower = question.lower()
        
        # Check for coverage-related terms
        if any(term in question_lower for term in ["cover", "coverage", "protect", "protection", "limit"]):
            return "coverage"
        
        # Check for exclusion-related terms
        if any(term in question_lower for term in ["exclude", "exclusion", "not cover", "exception", "exempt"]):
            return "exclusion"
        
        # Check for premium-related terms
        if any(term in question_lower for term in ["premium", "cost", "price", "payment", "pay"]):
            return "premium"
        
        # Default to general
        return "general"
