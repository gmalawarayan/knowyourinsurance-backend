"""
Insurance Policy Analysis Features Module

This module provides specialized analysis features for insurance policies:
- Coverage gap detection
- Policy comparison
- Risk assessment
- Premium analysis
"""

import logging
from typing import List, Dict, Any, Optional
import json

from app.core.llm_query import LLMQuerySystem
from app.core.vector_store import VectorStore
from langchain_community.llms import Ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InsuranceAnalyzer:
    """
    Provides specialized insurance policy analysis features.
    """
    
    def __init__(self, llm_query_system: LLMQuerySystem, vector_store: VectorStore):
        """
        Initialize the insurance analyzer.
        
        Args:
            llm_query_system: LLM query system for analysis
            vector_store: Vector store for document retrieval
        """
        self.llm_query = llm_query_system
        self.vector_store = vector_store

        self.llm = Ollama(model="llama3")

        logger.info("Initialized InsuranceAnalyzer")
    
    def analyze_coverage_gaps(self, document_id: str) -> Dict[str, Any]:
        """
        Analyze coverage gaps in an insurance policy.
        
        Args:
            document_id: Document ID of the policy to analyze
            
        Returns:
            Dictionary with coverage gap analysis
        """
        logger.info(f"Analyzing coverage gaps for document: {document_id}")
        return self.llm_query.analyze_coverage_gaps(document_id)
    
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
        return self.llm_query.compare_policies(document_id_1, document_id_2)
    
    def assess_risk(self, document_id: str) -> Dict[str, Any]:
        """
        Assess the risk profile based on an insurance policy.
        
        Args:
            document_id: Document ID of the policy to analyze
            
        Returns:
            Dictionary with risk assessment
        """
        logger.info(f"Assessing risk for document: {document_id}")
        
        # Retrieve policy chunks
        chunks = self.vector_store.get_document_chunks(document_id)
        
        if not chunks:
            return {
                "assessment": "No policy document found with the provided ID.",
                "risk_factors": []
            }
        
        # Format context from chunks
        context = self._format_context(chunks)
        
        # Create risk assessment prompt
        risk_prompt = """You are an AI assistant specialized in analyzing insurance policies.
        
        Assess the risk profile based on the following insurance policy:
        
        {context}
        
        Identify potential risk factors and provide a risk assessment. Consider:
        1. Coverage limits and whether they are adequate
        2. Deductible amounts and their impact on risk
        3. Exclusions that could leave the policyholder exposed
        4. Overall risk level (Low, Medium, High)
        
        Format your response as a JSON object with the following structure:
        {{
            "overall_risk_level": "Low/Medium/High",
            "summary": "Brief overall assessment of the policy's risk profile",
            "risk_factors": [
                {{
                    "factor": "Description of the risk factor",
                    "severity": "Low/Medium/High",
                    "recommendation": "Recommendation to address this risk"
                }},
                ...
            ]
        }}
        
        Risk Assessment:"""
        
        # Replace placeholder with actual context
        prompt = risk_prompt.replace("{context}", context)
        
        # Query LLM
        response = self.llm_query.llm.invoke(prompt)
        
        # Extract JSON from response
        try:
            # Try to parse the entire response as JSON
            assessment = json.loads(response)
        except json.JSONDecodeError:
            try:
                # Try to extract JSON from the response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    assessment = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, return the raw text
                assessment = {
                    "overall_risk_level": "Unknown",
                    "summary": "Risk assessment could not be properly formatted.",
                    "risk_factors": [],
                    "raw_assessment": response
                }
        
        return assessment
    
    def analyze_premium(self, document_id: str) -> Dict[str, Any]:
        """
        Analyze the premium structure of an insurance policy.
        
        Args:
            document_id: Document ID of the policy to analyze
            
        Returns:
            Dictionary with premium analysis
        """
        logger.info(f"Analyzing premium for document: {document_id}")
        
        # Retrieve policy chunks
        chunks = self.vector_store.get_document_chunks(document_id)
        
        if not chunks:
            return {
                "analysis": "No policy document found with the provided ID.",
                "factors": []
            }
        
        # Format context from chunks
        context = self._format_context(chunks)
        
        # Create premium analysis prompt
        premium_prompt = """You are an AI assistant specialized in analyzing insurance policies.
        
        Analyze the premium structure of the following insurance policy:
        
        {context}
        
        Provide a detailed analysis of the premium, including:
        1. Total premium amount
        2. Payment schedule or options
        3. Factors affecting the premium
        4. Potential ways to reduce the premium
        
        Format your response as a JSON object with the following structure:
        {{
            "total_premium": "Total premium amount (or 'Not specified' if not found)",
            "payment_schedule": "Payment schedule details (or 'Not specified' if not found)",
            "summary": "Brief overall assessment of the premium structure",
            "factors": [
                {{
                    "factor": "Description of a factor affecting the premium",
                    "impact": "How this factor impacts the premium",
                    "optimization": "How this factor could be optimized to reduce costs"
                }},
                ...
            ]
        }}
        
        Premium Analysis:"""
        
        # Replace placeholder with actual context
        prompt = premium_prompt.replace("{context}", context)
        
        # Query LLM
        response = self.llm_query.llm.invoke(prompt)
        
        # Extract JSON from response
        try:
            # Try to parse the entire response as JSON
            analysis = json.loads(response)
        except json.JSONDecodeError:
            try:
                # Try to extract JSON from the response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    analysis = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, return the raw text
                analysis = {
                    "total_premium": "Unknown",
                    "payment_schedule": "Unknown",
                    "summary": "Premium analysis could not be properly formatted.",
                    "factors": [],
                    "raw_analysis": response
                }
        
        return analysis
    
    def identify_policy_benchmarks(self, document_id: str, policy_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare policy against industry benchmarks.
        
        Args:
            document_id: Document ID of the policy to analyze
            policy_type: Optional policy type override
            
        Returns:
            Dictionary with benchmark comparison
        """
        logger.info(f"Identifying policy benchmarks for document: {document_id}")
        
        # Retrieve policy chunks
        chunks = self.vector_store.get_document_chunks(document_id)
        
        if not chunks:
            return {
                "analysis": "No policy document found with the provided ID.",
                "benchmarks": []
            }
        
        # Format context from chunks
        context = self._format_context(chunks)
        
        # Determine policy type if not provided
        if not policy_type:
            # Extract policy type from metadata if available
            for chunk in chunks:
                if "policy_type" in chunk.get("metadata", {}):
                    policy_type = chunk["metadata"]["policy_type"]
                    break
            
            # If still not found, use a prompt to determine policy type
            if not policy_type:
                type_prompt = """You are an AI assistant specialized in analyzing insurance policies.
                
                Based on the following insurance policy excerpt, determine the type of insurance policy:
                
                {context}
                
                Respond with ONLY ONE of the following policy types:
                - Auto Insurance
                - Home Insurance
                - Renters Insurance
                - Life Insurance
                - Health Insurance
                - Business Insurance
                - Umbrella Insurance
                - Travel Insurance
                
                Policy Type:"""
                
                # Replace placeholder with actual context
                prompt = type_prompt.replace("{context}", context)
                
                # Query LLM
                policy_type = self.llm_query.llm.invoke(prompt).strip()
        
        # Create benchmark comparison prompt
        benchmark_prompt = """You are an AI assistant specialized in analyzing insurance policies.
        
        Compare the following {policy_type} policy against industry benchmarks:
        
        {context}
        
        Provide a detailed comparison of how this policy compares to industry standards and benchmarks for {policy_type} policies. Consider:
        1. Coverage limits compared to recommended levels
        2. Deductibles compared to industry averages
        3. Exclusions that are unusual or differ from standard policies
        4. Premium cost compared to market averages
        
        Format your response as a JSON object with the following structure:
        {{
            "policy_type": "{policy_type}",
            "summary": "Brief overall assessment of how the policy compares to benchmarks",
            "benchmarks": [
                {{
                    "category": "Category name (e.g., Coverage Limits, Deductibles)",
                    "policy_value": "Value from the analyzed policy",
                    "benchmark_value": "Industry benchmark or recommended value",
                    "assessment": "How the policy compares to the benchmark (Below/Meets/Exceeds)"
                }},
                ...
            ]
        }}
        
        Benchmark Comparison:"""
        
        # Replace placeholders with actual values
        prompt = benchmark_prompt.replace("{policy_type}", policy_type).replace("{context}", context)
        
        # Query LLM
        response = self.llm_query.llm.invoke(prompt)
        
        # Extract JSON from response
        try:
            # Try to parse the entire response as JSON
            comparison = json.loads(response)
        except json.JSONDecodeError:
            try:
                # Try to extract JSON from the response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    comparison = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, return the raw text
                comparison = {
                    "policy_type": policy_type,
                    "summary": "Benchmark comparison could not be properly formatted.",
                    "benchmarks": [],
                    "raw_comparison": response
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
