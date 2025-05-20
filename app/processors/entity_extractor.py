"""
Insurance Entity Extraction Module

This module provides specialized extraction of insurance-specific entities from policy documents.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InsuranceEntityExtractor:
    """
    Extracts insurance-specific entities from policy documents, including:
    - Policy numbers
    - Coverage limits
    - Deductibles
    - Policy periods
    - Insured information
    """
    
    def __init__(self):
        """Initialize the insurance entity extractor."""
        logger.info("Initialized InsuranceEntityExtractor")
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract all insurance entities from document text.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {
            "policy_number": self._extract_policy_number(text),
            "policy_period": self._extract_policy_period(text),
            "insured": self._extract_insured_info(text),
            "coverage_limits": self._extract_coverage_limits(text),
            "deductibles": self._extract_deductibles(text),
            "premium": self._extract_premium(text),
            "exclusions": self._extract_exclusions(text),
            "policy_type": self._determine_policy_type(text),
        }
        
        return entities
    
    def _extract_policy_number(self, text: str) -> Optional[str]:
        """
        Extract policy number from text.
        
        Args:
            text: Document text
            
        Returns:
            Extracted policy number or None
        """
        patterns = [
            r'(?i)policy\s+number[:\s]+([A-Z0-9-]{5,20})',
            r'(?i)policy\s+no\.?[:\s]+([A-Z0-9-]{5,20})',
            r'(?i)policy\s+#[:\s]*([A-Z0-9-]{5,20})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_policy_period(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract policy period from text.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with start_date and end_date
        """
        result = {"start_date": None, "end_date": None}
        
        # Date patterns (various formats)
        date_pattern = r'(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12][0-9]|3[01])[\/\-](20\d{2})'
        
        # Look for policy period
        period_patterns = [
            rf'(?i)policy\s+period[:\s]+(from|)?\s*({date_pattern})\s+(to|through)\s+({date_pattern})',
            rf'(?i)effective\s+date[:\s]+({date_pattern}).*?expiration\s+date[:\s]+({date_pattern})',
        ]
        
        for pattern in period_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                # Find the date groups
                dates = [g for g in groups if re.match(date_pattern, g)]
                if len(dates) >= 2:
                    result["start_date"] = dates[0]
                    result["end_date"] = dates[1]
                    break
        
        return result
    
    def _extract_insured_info(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract insured information from text.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with insured information
        """
        result = {
            "name": None,
            "address": None,
            "phone": None,
            "email": None
        }
        
        # Extract insured name
        name_patterns = [
            r'(?i)named\s+insured[:\s]+([A-Za-z0-9\s\.,&]+?)(?:\n|$)',
            r'(?i)insured[:\s]+([A-Za-z0-9\s\.,&]+?)(?:\n|$)',
            r'(?i)policyholder[:\s]+([A-Za-z0-9\s\.,&]+?)(?:\n|$)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                result["name"] = match.group(1).strip()
                break
        
        # Extract address - look for address near insured name
        if result["name"]:
            address_search_area = text[text.find(result["name"]):text.find(result["name"]) + 500]
            address_pattern = r'(?i)(?:address|location)[:\s]+([A-Za-z0-9\s\.,#-]+(?:Road|Street|Avenue|Lane|Drive|Blvd|Boulevard|Ave|St|Rd|Ln|Dr)[A-Za-z0-9\s\.,#-]+)'
            
            address_match = re.search(address_pattern, address_search_area)
            if address_match:
                result["address"] = address_match.group(1).strip()
        
        # Extract phone number
        phone_pattern = r'(?i)(?:phone|telephone|mobile)[:\s]+(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            result["phone"] = phone_match.group(1).strip()
        
        # Extract email
        email_pattern = r'(?i)(?:email|e-mail)[:\s]+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        email_match = re.search(email_pattern, text)
        if email_match:
            result["email"] = email_match.group(1).strip()
        
        return result
    
    def _extract_coverage_limits(self, text: str) -> Dict[str, str]:
        """
        Extract coverage limits from text.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of coverage types and their limits
        """
        coverage_limits = {}
        
        # Common coverage types
        coverage_types = [
            "Bodily Injury Liability",
            "Property Damage Liability",
            "Personal Injury Protection",
            "Uninsured Motorist",
            "Underinsured Motorist",
            "Comprehensive",
            "Collision",
            "Medical Payments",
            "Dwelling",
            "Personal Property",
            "Liability",
            "Loss of Use",
            "Personal Liability",
            "Medical Expense",
        ]
        
        # Money pattern
        money_pattern = r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)'
        
        # Look for coverage section
        coverage_section_match = re.search(r'(?i)(?:coverage|limits)[s]?[:\s]+(.+?)(?:\n\n|\n[A-Z])', text, re.DOTALL)
        coverage_section = text
        if coverage_section_match:
            coverage_section = coverage_section_match.group(1)
        
        for coverage in coverage_types:
            # Create pattern for this coverage type
            pattern = rf'(?i){re.escape(coverage)}[:\s]+({money_pattern})'
            match = re.search(pattern, coverage_section)
            
            if match:
                coverage_limits[coverage] = match.group(1)
            else:
                # Try in the whole document if not found in coverage section
                match = re.search(pattern, text)
                if match:
                    coverage_limits[coverage] = match.group(1)
        
        # Look for any other coverage amounts with dollar signs
        if not coverage_limits:
            generic_coverage_pattern = rf'(?i)([A-Za-z\s]+)[:\s]+({money_pattern})'
            for match in re.finditer(generic_coverage_pattern, coverage_section):
                coverage_type = match.group(1).strip()
                amount = match.group(2)
                if coverage_type and amount and len(coverage_type) > 3:  # Avoid short matches
                    coverage_limits[coverage_type] = amount
        
        return coverage_limits
    
    def _extract_deductibles(self, text: str) -> Dict[str, str]:
        """
        Extract deductibles from text.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of coverage types and their deductibles
        """
        deductibles = {}
        
        # Look for deductible section
        deductible_section_match = re.search(r'(?i)deductible[s]?[:\s]+(.+?)(?:\n\n|\n[A-Z])', text, re.DOTALL)
        
        if deductible_section_match:
            deductible_section = deductible_section_match.group(1)
            
            # Money pattern
            money_pattern = r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)'
            
            # Look for coverage-specific deductibles
            coverage_deductible_matches = re.finditer(
                rf'(?i)([A-Za-z\s]+)[:\s]+({money_pattern})',
                deductible_section
            )
            
            for match in coverage_deductible_matches:
                coverage = match.group(1).strip()
                amount = match.group(2)
                deductibles[coverage] = amount
            
            # If no specific deductibles found, look for general deductible
            if not deductibles:
                general_match = re.search(money_pattern, deductible_section)
                if general_match:
                    deductibles["General"] = general_match.group(1)
        else:
            # Try to find deductibles near coverage mentions
            money_pattern = r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)'
            deductible_pattern = rf'(?i)(?:deductible|retention)[:\s]+({money_pattern})'
            
            match = re.search(deductible_pattern, text)
            if match:
                deductibles["General"] = match.group(1)
        
        return deductibles
    
    def _extract_premium(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract premium information from text.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with premium information
        """
        result = {
            "total": None,
            "payment_schedule": None,
            "breakdown": {}
        }
        
        # Extract total premium
        total_patterns = [
            r'(?i)total\s+premium[:\s]+\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)',
            r'(?i)premium[:\s]+\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)',
            r'(?i)total\s+cost[:\s]+\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)',
        ]
        
        for pattern in total_patterns:
            match = re.search(pattern, text)
            if match:
                result["total"] = match.group(1).strip()
                break
        
        # Extract payment schedule
        schedule_patterns = [
            r'(?i)payment\s+schedule[:\s]+([^\n]+)',
            r'(?i)payment\s+plan[:\s]+([^\n]+)',
            r'(?i)installment\s+plan[:\s]+([^\n]+)',
        ]
        
        for pattern in schedule_patterns:
            match = re.search(pattern, text)
            if match:
                result["payment_schedule"] = match.group(1).strip()
                break
        
        # Look for premium breakdown section
        premium_section_match = re.search(r'(?i)premium\s+breakdown[:\s]+(.+?)(?:\n\n|\n[A-Z])', text, re.DOTALL)
        
        if premium_section_match:
            premium_section = premium_section_match.group(1)
            
            # Money pattern
            money_pattern = r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)'
            
            # Look for coverage-specific premiums
            coverage_premium_matches = re.finditer(
                rf'(?i)([A-Za-z\s]+)[:\s]+({money_pattern})',
                premium_section
            )
            
            for match in coverage_premium_matches:
                coverage = match.group(1).strip()
                amount = match.group(2)
                result["breakdown"][coverage] = amount
        
        return result
    
    def _extract_exclusions(self, text: str) -> List[str]:
        """
        Extract policy exclusions from text.
        
        Args:
            text: Document text
            
        Returns:
            List of exclusion statements
        """
        exclusions = []
        
        # Look for exclusions section
        exclusion_section_match = re.search(
            r'(?i)(?:exclusions|what\s+is\s+not\s+covered)[:\s]+(.+?)(?:\n\n\n|\n[A-Z][A-Z])',
            text,
            re.DOTALL
        )
        
        if exclusion_section_match:
            exclusion_section = exclusion_section_match.group(1)
            
            # Split by bullet points or numbers
            bullet_splits = re.split(r'(?:\n\s*[•\-\*]|\n\s*\d+\.)', exclusion_section)
            
            for split in bullet_splits:
                clean_split = split.strip()
                if clean_split and len(clean_split) > 20:  # Avoid short fragments
                    exclusions.append(clean_split)
        
        return exclusions
    
    def _determine_policy_type(self, text: str) -> str:
        """
        Determine the type of insurance policy.
        
        Args:
            text: Document text
            
        Returns:
            Policy type string
        """
        # Common policy types and their indicators
        policy_types = {
            "Auto Insurance": [
                r'(?i)auto(?:mobile)?\s+(?:insurance|policy)',
                r'(?i)motor\s+vehicle\s+(?:insurance|policy)',
                r'(?i)car\s+(?:insurance|policy)',
                r'(?i)collision\s+coverage',
                r'(?i)comprehensive\s+coverage',
                r'(?i)bodily\s+injury\s+liability',
                r'(?i)property\s+damage\s+liability',
            ],
            "Home Insurance": [
                r'(?i)home(?:owner)?s?\s+(?:insurance|policy)',
                r'(?i)dwelling\s+coverage',
                r'(?i)personal\s+property\s+coverage',
                r'(?i)structure\s+coverage',
            ],
            "Renters Insurance": [
                r'(?i)renter[\'s]?\s+(?:insurance|policy)',
                r'(?i)tenant[\'s]?\s+(?:insurance|policy)',
            ],
            "Life Insurance": [
                r'(?i)life\s+(?:insurance|policy)',
                r'(?i)term\s+life',
                r'(?i)whole\s+life',
                r'(?i)universal\s+life',
                r'(?i)death\s+benefit',
            ],
            "Health Insurance": [
                r'(?i)health\s+(?:insurance|policy|plan)',
                r'(?i)medical\s+(?:insurance|policy|plan)',
                r'(?i)copay',
                r'(?i)coinsurance',
                r'(?i)deductible',
                r'(?i)out-of-pocket\s+maximum',
            ],
            "Business Insurance": [
                r'(?i)business\s+(?:insurance|policy)',
                r'(?i)commercial\s+(?:insurance|policy)',
                r'(?i)general\s+liability',
                r'(?i)professional\s+liability',
                r'(?i)business\s+interruption',
            ],
            "Umbrella Insurance": [
                r'(?i)umbrella\s+(?:insurance|policy)',
                r'(?i)excess\s+liability',
            ],
            "Travel Insurance": [
                r'(?i)travel\s+(?:insurance|policy)',
                r'(?i)trip\s+cancellation',
                r'(?i)baggage\s+loss',
            ],
        }
        
        # Count matches for each policy type
        type_scores = {policy_type: 0 for policy_type in policy_types}
        
        for policy_type, patterns in policy_types.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                type_scores[policy_type] += len(matches)
        
        # Find the policy type with the highest score
        max_score = 0
        detected_type = "Unknown"
        
        for policy_type, score in type_scores.items():
            if score > max_score:
                max_score = score
                detected_type = policy_type
        
        return detected_type
