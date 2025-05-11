"""
Insurance Entity Extraction Module

Uses regular expressions to extract common insurance entities.
Can be expanded with more sophisticated NLP techniques (e.g., NER models).
"""

import re
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InsuranceEntityExtractor:
    """
    Extracts key entities from insurance policy text using regex.
    """

    def __init__(self):
        """
        Initialize the entity extractor with predefined regex patterns.
        """
        self.patterns = {
            # Policy Number (various common formats)
            "policy_number": r"Policy\s*(?:Number|No\.?)\s*:?\s*([A-Za-z0-9\-]+)",
            # Effective Date (common date formats)
            "effective_date": r"Effective\s*(?:Date)?\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+\s+\d{1,2},\s+\d{4})",
            # Expiration Date
            "expiration_date": r"Expiration\s*(?:Date)?\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+\s+\d{1,2},\s+\d{4})",
            # Insured Name (looks for lines starting with Insured or Named Insured)
            "insured_name": r"(?:Named\s+)?Insured\s*:?\s*(.*?)(?:\n|Address|$)",
            # Total Premium (looks for currency amounts near 'Total Premium')
            "total_premium": r"Total\s*(?:Annual\s*)?Premium\s*:?\s*\$?([\d,]+\.\d{2})",
            # Deductible (looks for currency amounts near 'Deductible')
            "deductible": r"Deductible\s*:?\s*\$?([\d,]+\.\d{2})",
            # Liability Limit (looks for currency amounts near 'Liability')
            "liability_limit": r"(?:Bodily\s+Injury\s+)?Liability\s*(?:Limit)?\s*:?\s*\$?([\d,]+(?:/\$[\d,]+)?)",
            # Policy Type (simple check for common types)
            "policy_type": r"(Auto|Homeowners|Renters|Life|Health|Business|Umbrella|Travel)\s+Insurance\s+Policy"
        }
        logger.info("Initialized InsuranceEntityExtractor with regex patterns.")

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from the provided text using regex patterns.

        Args:
            text: The text content of the insurance policy.

        Returns:
            A dictionary containing extracted entities.
        """
        entities = {}
        logger.info("Starting entity extraction.")

        for entity_name, pattern in self.patterns.items():
            try:
                # Use re.IGNORECASE for case-insensitive matching
                # Use re.MULTILINE for patterns involving start/end of lines (^/$)
                flags = re.IGNORECASE | re.MULTILINE
                matches = re.findall(pattern, text, flags)

                if matches:
                    # Store the first match, cleaning whitespace
                    # For patterns with multiple capture groups, findall returns tuples
                    first_match = matches[0]
                    if isinstance(first_match, tuple):
                        # Find the first non-empty group if multiple capture groups exist
                        cleaned_match = next((group.strip() for group in first_match if group and group.strip()), None)
                    else:
                        cleaned_match = first_match.strip()

                    if cleaned_match:
                        entities[entity_name] = cleaned_match
                        logger.debug(f"Found entity '{entity_name}': {cleaned_match}")
                    else:
                        logger.debug(f"Found match for '{entity_name}', but capture group was empty.")
                else:
                    logger.debug(f"No match found for entity '{entity_name}'.")

            except Exception as e:
                logger.error(f"Error processing regex for entity '{entity_name}': {str(e)}")

        # Post-processing and refinement can be added here
        # e.g., standardize date formats, clean currency values
        if 'total_premium' in entities:
            entities['total_premium'] = entities['total_premium'].replace(',', '')
        if 'deductible' in entities:
            entities['deductible'] = entities['deductible'].replace(',', '')

        logger.info(f"Entity extraction complete. Found {len(entities)} entities.")
        
        return entities