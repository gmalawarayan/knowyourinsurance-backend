"""
Enhanced Risk Assessment Module for Insurance Policy Analyzer
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RiskAssessmentAnalyzer:
    """
    Specialized analyzer for assessing risks in insurance policies.
    """
    
    def __init__(self, llm_query_system):
        """Initialize with the LLM query system."""
        self.llm_query = llm_query_system
    
    def analyze_risk_assessment(self, document_id: str, document_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a comprehensive risk assessment on an insurance policy.
        
        Args:
            document_id: The ID of the document
            document_text: The full text of the policy document
            metadata: Extracted metadata about the policy
            
        Returns:
            Dictionary containing risk assessment results
        """
        logger.info(f"Performing risk assessment for document {document_id}")
        
        # Extract policy type from metadata if available
        policy_type = metadata.get("policy_type", "Unknown")
        
        # Create a specialized prompt with insurance domain knowledge
        prompt = f"""
        You are an expert insurance risk analyst with deep knowledge of {policy_type} policies.
        
        I'm going to provide you with an insurance policy document. Your task is to perform a comprehensive 
        risk assessment, identifying areas where the policyholder may be exposed to financial or legal risks.
        
        Analyze the following aspects:
        
        1. Overall Risk Score: Rate the policy's overall risk level on a scale of 1-10 (1 being minimal risk, 10 being severe risk)
        2. Critical Exposures: Identify the 3-5 most significant areas where the policyholder is exposed to risk
        3. Risk Factors: List specific factors that contribute to the policyholder's risk profile
        4. Mitigation Recommendations: Provide specific recommendations to address each identified risk
        5. Industry Comparison: How does this policy compare to industry standards for similar coverage?
        
        Format your response as a JSON object with the following structure:
        {{
            "overall_risk_score": number,
            "critical_exposures": [list of strings],
            "risk_factors": [list of strings],
            "mitigation_recommendations": [list of strings],
            "industry_comparison": string
        }}
        
        Here is the policy document:
        {document_text[:5000]}  # Using first 5000 chars for context
        """
        
        # Query the LLM with the specialized prompt
        try:
            response = self.llm_query.query(prompt)
            
            # Process the response to extract the risk assessment
            import json
            try:
                # Try to parse as JSON
                assessment = json.loads(response)
                return self._validate_assessment(assessment)
            except json.JSONDecodeError:
                # If not valid JSON, extract structured data from text
                return self._extract_assessment_from_text(response)
                
        except Exception as e:
            logger.error(f"Error performing risk assessment: {str(e)}")
            return {
                "overall_risk_score": 5,
                "critical_exposures": ["Error performing risk assessment. Please try again."],
                "risk_factors": [],
                "mitigation_recommendations": [],
                "industry_comparison": "Unable to compare due to analysis error."
            }
    
    def _validate_assessment(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the assessment data."""
        result = {
            "overall_risk_score": 5,
            "critical_exposures": [],
            "risk_factors": [],
            "mitigation_recommendations": [],
            "industry_comparison": ""
        }
        
        # Copy valid fields from the assessment
        if "overall_risk_score" in assessment and isinstance(assessment["overall_risk_score"], (int, float)):
            result["overall_risk_score"] = min(max(int(assessment["overall_risk_score"]), 1), 10)
            
        if "critical_exposures" in assessment and isinstance(assessment["critical_exposures"], list):
            result["critical_exposures"] = assessment["critical_exposures"]
            
        if "risk_factors" in assessment and isinstance(assessment["risk_factors"], list):
            result["risk_factors"] = assessment["risk_factors"]
            
        if "mitigation_recommendations" in assessment and isinstance(assessment["mitigation_recommendations"], list):
            result["mitigation_recommendations"] = assessment["mitigation_recommendations"]
            
        if "industry_comparison" in assessment and isinstance(assessment["industry_comparison"], str):
            result["industry_comparison"] = assessment["industry_comparison"]
            
        return result
    
    def _extract_assessment_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured risk assessment from non-JSON text response."""
        result = {
            "overall_risk_score": 5,
            "critical_exposures": [],
            "risk_factors": [],
            "mitigation_recommendations": [],
            "industry_comparison": ""
        }
        
        # Try to extract overall risk score
        import re
        risk_score_match = re.search(r'(?:overall|risk)\s+(?:risk|score)(?:\s+is)?(?:\s*[:=]\s*)?(\d+)', text, re.IGNORECASE)
        if risk_score_match:
            try:
                score = int(risk_score_match.group(1))
                result["overall_risk_score"] = min(max(score, 1), 10)
            except ValueError:
                pass
        
        # Extract sections based on headers
        sections = {
            "critical_exposures": ["critical exposures", "significant exposures", "key exposures"],
            "risk_factors": ["risk factors", "factors", "risks identified"],
            "mitigation_recommendations": ["mitigation", "recommendations", "suggested actions"],
            "industry_comparison": ["industry comparison", "industry standard", "compared to industry"]
        }
        
        for key, headers in sections.items():
            for header in headers:
                pattern = f"(?:{header})(?:\s*[:=]\s*)(.*?)(?=(?:{'|'.join(sum(sections.values(), []))})|$)"
                matches = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if matches:
                    content = matches.group(1).strip()
                    if key == "industry_comparison":
                        result[key] = content
                    else:
                        # Extract list items
                        items = self._extract_list_items(content)
                        if items:
                            result[key] = items
                    break
        
        return result
    
    def _extract_list_items(self, text: str) -> list:
        """Extract list items from text."""
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
        
        # If all else fails, return the whole text as one item
        return [text.strip()] if text.strip() else []
