"""
Chunking Module for Insurance Policy Analysis

This module provides specialized chunking strategies for insurance policy documents.
"""

from typing import List, Dict, Any, Optional
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolicyChunker:
    """
    Specialized chunker for insurance policy documents that preserves
    semantic structure and important policy sections.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 preserve_sections: bool = True):
        """
        Initialize the policy chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            preserve_sections: Whether to try to keep policy sections intact
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_sections = preserve_sections
        logger.info(f"Initialized PolicyChunker with chunk_size={chunk_size}, "
                   f"chunk_overlap={chunk_overlap}, preserve_sections={preserve_sections}")
    
    def chunk_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split document into chunks optimized for insurance policy analysis.
        
        Args:
            text: Document text
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
            
        if metadata is None:
            metadata = {}
            
        # If preserve_sections is enabled, try to split by policy sections first
        if self.preserve_sections:
            sections = self._identify_policy_sections(text)
            if sections:
                return self._chunk_by_sections(sections, metadata)
        
        # Fall back to standard chunking if no sections or preserve_sections is disabled
        return self._chunk_by_size(text, metadata)
    
    def _identify_policy_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify standard insurance policy sections.
        
        Args:
            text: Document text
            
        Returns:
            List of section dictionaries with title and content
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
        
        # If no sections found, return empty list
        if not matches:
            return []
        
        sections = []
        
        # Add first section (before first header)
        if matches[0].start() > 0:
            sections.append({
                "title": "INTRODUCTION",
                "content": text[:matches[0].start()].strip(),
                "start_char": 0,
                "end_char": matches[0].start()
            })
        
        # Add middle sections
        for i in range(len(matches)):
            section_title = matches[i].group(1).strip()
            start_pos = matches[i].end()
            
            # End position is start of next section or end of text
            end_pos = matches[i+1].start() if i < len(matches) - 1 else len(text)
            
            section_content = text[start_pos:end_pos].strip()
            sections.append({
                "title": section_title,
                "content": section_content,
                "start_char": start_pos,
                "end_char": end_pos
            })
        
        return sections
    
    def _chunk_by_sections(self, sections: List[Dict[str, Any]], 
                          base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunks based on document sections.
        
        Args:
            sections: List of section dictionaries
            base_metadata: Base metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        for section in sections:
            section_title = section["title"]
            section_content = section["content"]
            
            # If section is small enough, keep it as a single chunk
            if len(section_content) <= self.chunk_size:
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "section": section_title,
                    "start_char": section["start_char"],
                    "end_char": section["end_char"],
                    "is_complete_section": True
                })
                
                chunks.append({
                    "text": section_content,
                    "metadata": chunk_metadata
                })
                continue
            
            # Otherwise, split the section into overlapping chunks
            section_chunks = self._chunk_text(
                section_content, 
                section_title,
                section["start_char"],
                base_metadata
            )
            
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_by_size(self, text: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunks based on size without considering sections.
        
        Args:
            text: Document text
            base_metadata: Base metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries
        """
        return self._chunk_text(text, "DOCUMENT", 0, base_metadata)
    
    def _chunk_text(self, text: str, section_title: str, 
                   start_offset: int, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks of specified size.
        
        Args:
            text: Text to chunk
            section_title: Section title for metadata
            start_offset: Character offset from document start
            base_metadata: Base metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # Try to split on paragraph boundaries first
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        current_start = start_offset
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph exceeds chunk size and we already have content,
            # save the current chunk and start a new one
            if current_chunk and len(current_chunk) + len(paragraph) > self.chunk_size:
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "section": section_title,
                    "start_char": current_start,
                    "end_char": current_start + len(current_chunk),
                    "is_complete_section": False
                })
                
                chunks.append({
                    "text": current_chunk,
                    "metadata": chunk_metadata
                })
                
                # Calculate overlap - try to keep whole paragraphs
                overlap_text = ""
                overlap_paragraphs = current_chunk.split("\n\n")
                
                # Take the last few paragraphs that fit within overlap size
                for p in reversed(overlap_paragraphs):
                    if len(p) + len(overlap_text) <= self.chunk_overlap:
                        overlap_text = p + "\n\n" + overlap_text
                    else:
                        break
                
                # If we couldn't get enough paragraphs, take the last part of the text
                if not overlap_text or len(overlap_text) < self.chunk_overlap / 2:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                
                current_chunk = overlap_text
                current_start = current_start + len(current_chunk) - len(overlap_text)
            
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        
        # Add the final chunk if there's anything left
        if current_chunk:
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "section": section_title,
                "start_char": current_start,
                "end_char": current_start + len(current_chunk),
                "is_complete_section": False
            })
            
            chunks.append({
                "text": current_chunk,
                "metadata": chunk_metadata
            })
        
        return chunks
