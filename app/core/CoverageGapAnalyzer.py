"""
Enhanced Coverage Gap Analysis Module for Insurance Policy Analyzer
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class CoverageGapAnalyzer:
    """
    Specialized analyzer for identifying coverage gaps in insurance policies.
    """
    
    def __init__(self, llm_query_system):
        """Initialize with the LLM query system."""
        self.llm_query = llm_query_system
    
    def analyze_coverage_gaps(self, document_id: str, document_text: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Analyze an insurance policy for coverage gaps.
        
        Args:
            document_id: The ID of the document
            document_text: The full text of the policy document
            metadata: Extracted metadata about the policy
            
        Returns:
            List of identified coverage gaps
        """
        logger.info(f"Analyzing coverage gaps for document {document_id}")
        
        # Extract policy type from metadata if available
        policy_type = metadata.get("policy_type", "Unknown")
        
        # Create a specialized prompt with insurance domain knowledge
        prompt = f"""
        You are an expert insurance analyst specializing in identifying coverage gaps.
        
        I'm going to provide you with an insurance policy document. Your task is to identify specific coverage gaps 
        or exclusions that might leave the policyholder vulnerable. Focus on:
        
        1. Standard coverages missing from this {policy_type} policy
        2. Low coverage limits relative to industry standards
        3. Exclusions that could create significant exposure
        4. Conditions that might void coverage in common scenarios
        5. Missing endorsements that are typically recommended
        
        For each gap, provide a clear, specific description of:
        - What coverage is missing or insufficient
        - Why this creates risk for the policyholder
        - What standard in the industry would typically cover this
        
        Format your response as a JSON list of strings, each describing a specific coverage gap.
        Only include actual gaps - if the policy appears comprehensive, return an empty list.
        
        Here is the policy document:
        {document_text[:5000]}  # Using first 5000 chars for context
        """
        
        # Query the LLM with the specialized prompt
        try:
            response = self.llm_query.query(prompt)
            
            # Process the response to extract the list of gaps
            import json
            try:
                # Try to parse as JSON
                gaps = json.loads(response)
                if isinstance(gaps, list):
                    return gaps
                else:
                    # If not a list, try to extract from the response
                    return self._extract_gaps_from_text(response)
            except json.JSONDecodeError:
                # If not valid JSON, extract gaps from text
                return self._extract_gaps_from_text(response)
                
        except Exception as e:
            logger.error(f"Error analyzing coverage gaps: {str(e)}")
            return ["Error analyzing coverage gaps. Please try again."]
    
    def _extract_gaps_from_text(self, text: str) -> List[str]:
        """Extract coverage gaps from non-JSON text response."""
        # Look for numbered or bulleted lists
        import re
        
        # Try to find numbered items (1. Item)
        numbered_items = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', text, re.DOTALL)
        if numbered_items:
            return [item.strip() for item in numbered_items if item.strip()]
        
        # Try to find bulleted items (- Item or • Item)
        bulleted_items = re.findall(r'[-•]\s+(.*?)(?=[-•]|$)', text, re.DOTALL)
        if bulleted_items:
            return [item.strip() for item in bulleted_items if item.strip()]
        
        # If no structured list found, split by newlines and filter
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) > 1:
            return lines
        
        # If all else fails, return the whole text as one gap
        return [text.strip()] if text.strip() else []
