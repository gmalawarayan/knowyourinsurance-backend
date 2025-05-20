"""
Document Processing Module for Insurance Policy Analysis

This module handles the extraction, cleaning, and chunking of insurance policy documents.
"""

import os
import re
import fitz  # PyMuPDF
import pdfplumber
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles the processing of insurance policy documents, including:
    - Text extraction from PDFs
    - Document cleaning and normalization
    - Chunking for vector storage
    - Basic insurance entity recognition
    """
    
    def __init__(self, use_ocr: bool = False):
        """
        Initialize the document processor.
        
        Args:
            use_ocr: Whether to use OCR for image-based PDFs
        """
        self.use_ocr = use_ocr
        logger.info(f"Initialized DocumentProcessor with OCR set to {use_ocr}")
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document file and return extracted content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing processed document data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            text, metadata = self._extract_from_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Check if this is likely an insurance policy
        if not self._is_insurance_policy(text):
            logger.warning(f"Document does not appear to be an insurance policy: {file_path}")
            return {
                "is_insurance_policy": False,
                "text": "",
                "metadata": metadata,
                "chunks": [],
                "entities": {}
            }
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Split into chunks
        chunks = self._chunk_document(cleaned_text)
        
        # Extract insurance-specific entities
        entities = self._extract_insurance_entities(cleaned_text)
        
        return {
            "is_insurance_policy": True,
            "text": cleaned_text,
            "metadata": metadata,
            "chunks": chunks,
            "entities": entities
        }
    
    def _extract_from_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata from a PDF file.
        Tries PyMuPDF first, falls back to pdfplumber for complex PDFs.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted text, metadata dictionary)
        """
        logger.info(f"Extracting text from PDF: {file_path}")
        
        # Try PyMuPDF first (faster)
        try:
            doc = fitz.open(file_path)
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "page_count": len(doc),
            }
            
            full_text = ""
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}"
                elif self.use_ocr:
                    # If page has no text and OCR is enabled, we would use OCR here
                    # This is a placeholder for OCR implementation
                    logger.info(f"Page {page_num + 1} has no text, OCR would be used here")
                    pass
            
            doc.close()
            
            # If we got meaningful text, return it
            if len(full_text.strip()) > 100:
                return full_text, metadata
                
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed, falling back to pdfplumber: {str(e)}")
        
        # Fall back to pdfplumber (better for some complex PDFs)
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata = {
                    "page_count": len(pdf.pages),
                }
                
                full_text = ""
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
                
                # Try to extract tables if text is limited
                if len(full_text.strip()) < 1000:
                    for page_num, page in enumerate(pdf.pages):
                        tables = page.extract_tables()
                        for table_num, table in enumerate(tables):
                            if table:
                                full_text += f"\n--- Table {page_num + 1}.{table_num + 1} ---\n"
                                for row in table:
                                    full_text += " | ".join([str(cell or "") for cell in row]) + "\n"
                
                return full_text, metadata
                
        except Exception as e:
            logger.error(f"PDF extraction failed with both methods: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned and normalized text
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Fix common OCR errors in insurance policies
        text = text.replace('S', '$').replace('l00', '100')
        
        # Normalize newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Split document into overlapping chunks for processing.
        
        Args:
            text: Document text
            chunk_size: Target size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        
        # Split by natural section boundaries first
        sections = self._split_into_sections(text)
        
        current_chunk = ""
        current_section = ""
        
        for section_title, section_content in sections:
            current_section = section_title
            
            # If section is small enough, add it as a single chunk
            if len(section_content) < chunk_size:
                chunks.append({
                    "text": section_content,
                    "metadata": {
                        "section": section_title,
                        "start_char": len(current_chunk),
                        "end_char": len(current_chunk) + len(section_content)
                    }
                })
                current_chunk += section_content + " "
                continue
            
            # Otherwise, split the section into overlapping chunks
            words = section_content.split()
            current_chunk_words = []
            
            for word in words:
                current_chunk_words.append(word)
                
                # If we've reached the chunk size, save the chunk
                if len(" ".join(current_chunk_words)) >= chunk_size:
                    chunk_text = " ".join(current_chunk_words)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "section": section_title,
                            "start_char": len(current_chunk),
                            "end_char": len(current_chunk) + len(chunk_text)
                        }
                    })
                    
                    # Keep overlap words for next chunk
                    overlap_word_count = min(len(current_chunk_words), 
                                            max(20, int(overlap / 5)))  # Approximate words in overlap
                    current_chunk_words = current_chunk_words[-overlap_word_count:]
                    current_chunk += chunk_text + " "
            
            # Add any remaining words as the final chunk for this section
            if current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "section": section_title,
                        "start_char": len(current_chunk),
                        "end_char": len(current_chunk) + len(chunk_text)
                    }
                })
                current_chunk += chunk_text + " "
        
        return chunks
    
    def _split_into_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Split document into logical sections based on headers.
        
        Args:
            text: Document text
            
        Returns:
            List of (section_title, section_content) tuples
        """
        # Common insurance policy section headers
        section_patterns = [
            r'(?i)(?:\n|\r|\f)+(DECLARATIONS|POLICY\s+DECLARATIONS)',
            r'(?i)(?:\n|\r|\f)+(COVERAGE[S]?(?:\s+[A-Z])?)',
            r'(?i)(?:\n|\r|\f)+(EXCLUSIONS|WHAT\s+IS\s+NOT\s+COVERED)',
            r'(?i)(?:\n|\r|\f)+(CONDITIONS|POLICY\s+CONDITIONS)',
            r'(?i)(?:\n|\r|\f)+(DEFINITIONS|DEFINED\s+TERMS)',
            r'(?i)(?:\n|\r|\f)+(ENDORSEMENTS)',
            r'(?i)(?:\n|\r|\f)+(LIMITS\s+OF\s+LIABILITY)',
            r'(?i)(?:\n|\r|\f)+(DEDUCTIBLE[S]?)',
            r'(?i)(?:\n|\r|\f)+(PREMIUM\s+CALCULATION)',
        ]
        
        # Combine patterns
        combined_pattern = '|'.join(section_patterns)
        
        # Find all matches
        matches = list(re.finditer(combined_pattern, text))
        
        # If no sections found, return whole document as one section
        if not matches:
            return [("POLICY DOCUMENT", text)]
        
        sections = []
        
        # Add first section (before first header)
        if matches[0].start() > 0:
            sections.append(("INTRODUCTION", text[:matches[0].start()]))
        
        # Add middle sections
        for i in range(len(matches)):
            section_title = matches[i].group(1).strip()
            start_pos = matches[i].end()
            
            # End position is start of next section or end of text
            end_pos = matches[i+1].start() if i < len(matches) - 1 else len(text)
            
            section_content = text[start_pos:end_pos].strip()
            sections.append((section_title, section_content))
        
        return sections
    
    def _is_insurance_policy(self, text: str) -> bool:
        """
        Determine if the document is likely an insurance policy.
        
        Args:
            text: Document text
            
        Returns:
            Boolean indicating if document appears to be an insurance policy
        """
        # Common terms found in insurance policies
        insurance_terms = [
            r'(?i)policy\s+number',
            r'(?i)coverage',
            r'(?i)premium',
            r'(?i)deductible',
            r'(?i)insured',
            r'(?i)liability',
            r'(?i)exclusions?',
            r'(?i)endorsements?',
            r'(?i)declarations',
            r'(?i)insurance\s+company',
            r'(?i)policy\s+period',
            r'(?i)effective\s+date',
            r'(?i)expiration\s+date',
        ]
        
        # Count how many insurance terms are found
        term_count = 0
        for term in insurance_terms:
            if re.search(term, text):
                term_count += 1
        
        # If more than 5 terms are found, it's likely an insurance policy
        return term_count >= 5
    
    def _extract_insurance_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract insurance-specific entities from the document.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {
            "policy_number": self._extract_policy_number(text),
            "policy_period": self._extract_policy_period(text),
            "insured_name": self._extract_insured_name(text),
            "coverage_limits": self._extract_coverage_limits(text),
            "deductibles": self._extract_deductibles(text),
            "premium": self._extract_premium(text),
        }
        
        return entities
    
    def _extract_policy_number(self, text: str) -> Optional[str]:
        """Extract policy number from text"""
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
        """Extract policy period from text"""
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
    
    def _extract_insured_name(self, text: str) -> Optional[str]:
        """Extract insured name from text"""
        patterns = [
            r'(?i)named\s+insured[:\s]+([A-Za-z0-9\s\.,&]+?)(?:\n|$)',
            r'(?i)insured[:\s]+([A-Za-z0-9\s\.,&]+?)(?:\n|$)',
            r'(?i)policyholder[:\s]+([A-Za-z0-9\s\.,&]+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_coverage_limits(self, text: str) -> Dict[str, str]:
        """Extract coverage limits from text"""
        coverage_limits = {}
        
        # Common coverage types
        coverage_types = [
            "Bodily Injury Liability",
            "Property Damage Liability",
            "Personal Injury Protection",
            "Uninsured Motorist",
            "Comprehensive",
            "Collision",
            "Medical Payments",
            "Dwelling",
            "Personal Property",
            "Liability",
            "Loss of Use",
        ]
        
        # Money pattern
        money_pattern = r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)'
        
        for coverage in coverage_types:
            # Create pattern for this coverage type
            pattern = rf'(?i){re.escape(coverage)}[:\s]+({money_pattern})'
            match = re.search(pattern, text)
            
            if match:
                coverage_limits[coverage] = match.group(1)
        
        return coverage_limits
    
    def _extract_deductibles(self, text: str) -> Dict[str, str]:
        """Extract deductibles from text"""
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
        
        return deductibles
    
    def _extract_premium(self, text: str) -> Optional[str]:
        """Extract premium amount from text"""
        patterns = [
            r'(?i)total\s+premium[:\s]+\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)',
            r'(?i)premium[:\s]+\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)',
            r'(?i)total\s+cost[:\s]+\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return None
