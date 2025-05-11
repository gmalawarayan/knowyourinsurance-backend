"""
Document Processing Module (macOS compatible using PyPDF2)

Handles the processing of insurance policy documents, including:
- Text extraction from PDFs
- Document cleaning and normalization
- Basic insurance policy detection
"""

import os
import re
import PyPDF2
import pdfplumber
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles the processing of insurance policy documents.
    """

    def __init__(self, use_ocr: bool = False):
        """
        Initialize the document processor.

        Args:
            use_ocr: Whether to use OCR for image-based PDFs (Not fully implemented in this version)
        """
        self.use_ocr = use_ocr
        logger.info(f"Initialized DocumentProcessor (PyPDF2 version) with OCR set to {use_ocr}")

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

        # Clean and normalize text
        cleaned_text = self._clean_text(text)

        # Check if this is likely an insurance policy
        is_policy = self._is_insurance_policy(cleaned_text)

        # Basic entity extraction (can be expanded)
        entities = {}
        if is_policy:
            # Placeholder for entity extraction logic if needed here
            # For now, entity extraction is handled later in the pipeline
            pass

        return {
            "is_insurance_policy": is_policy,
            "text": cleaned_text if is_policy else "", # Return empty text if not a policy
            "metadata": metadata,
            "entities": entities # Basic entities extracted here
        }

    def _extract_from_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata from a PDF file using PyPDF2 and pdfplumber.

        Args:
            file_path: Path to the PDF file

        Returns:
            Tuple of (extracted text, metadata dictionary)
        """
        logger.info(f"Extracting text from PDF: {file_path}")
        full_text = ""
        metadata = {}

        # Try PyPDF2 first
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                meta = reader.metadata
                metadata = {
                    "title": meta.get('/Title', '') if meta else '',
                    "author": meta.get('/Author', '') if meta else '',
                    "subject": meta.get('/Subject', '') if meta else '',
                    "creator": meta.get('/Creator', '') if meta else '',
                    "producer": meta.get('/Producer', '') if meta else '',
                    "page_count": len(reader.pages),
                }

                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"

                # If PyPDF2 extracted substantial text, use it
                if len(full_text.strip()) > 200: # Heuristic: check if meaningful text was extracted
                    logger.info(f"Successfully extracted text using PyPDF2 for {file_path}")
                    return full_text, metadata
                else:
                    logger.warning(f"PyPDF2 extracted minimal text ({len(full_text.strip())} chars) for {file_path}. Trying pdfplumber.")
                    full_text = "" # Reset text if PyPDF2 failed

        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed for {file_path}: {str(e)}. Falling back to pdfplumber.")
            full_text = "" # Ensure text is reset

        # Fall back to pdfplumber if PyPDF2 failed or extracted minimal text
        try:
            with pdfplumber.open(file_path) as pdf:
                # Update metadata if pdfplumber provides more
                metadata["page_count"] = len(pdf.pages)
                if not metadata.get("title") and pdf.metadata.get("Title"):
                    metadata["title"] = pdf.metadata.get("Title")
                # Add other metadata fields if needed

                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"

                    # Optional: Extract tables if needed, especially if text extraction is poor
                    # tables = page.extract_tables()
                    # for table_num, table in enumerate(tables):
                    #     if table:
                    #         full_text += f"\n--- Table {page_num + 1}.{table_num + 1} ---\n"
                    #         for row in table:
                    #             full_text += " | ".join([str(cell or "") for cell in row]) + "\n"

            logger.info(f"Successfully extracted text using pdfplumber for {file_path}")
            return full_text, metadata

        except Exception as e:
            logger.error(f"PDF extraction failed with both PyPDF2 and pdfplumber for {file_path}: {str(e)}")
            # Return empty text and basic metadata if all fails
            return "", {"page_count": metadata.get("page_count", 0), "error": str(e)}


    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'(\n\s*)+', '\n', text).strip()
        # Add more cleaning rules as needed (e.g., remove headers/footers, special characters)
        return text

    def _is_insurance_policy(self, text: str) -> bool:
        """
        Check if the document text likely represents an insurance policy.
        Uses simple keyword heuristics.

        Args:
            text: Cleaned document text

        Returns:
            True if likely an insurance policy, False otherwise
        """
        # Simple heuristic based on common insurance terms
        keywords = [
            "insurance policy", "policy number", "coverage", "premium",
            "deductible", "liability", "declaration page", "insured",
            "underwriter", "claim"
        ]
        text_lower = text.lower()
        found_keywords = sum(1 for keyword in keywords if keyword in text_lower)

        # Require at least 3 keywords and check text length
        is_policy = found_keywords >= 3 and len(text) > 500
        if not is_policy:
            logger.info(f"Document determined not to be an insurance policy (Keywords: {found_keywords}, Length: {len(text)})")
        return is_policy

