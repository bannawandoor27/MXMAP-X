"""PDF text extraction and preprocessing."""

import logging
import re
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from .config import config


logger = logging.getLogger(__name__)


class PDFCleaner:
    """Extracts and cleans text from PDFs for LLM processing."""
    
    # Section headers to identify relevant content
    RELEVANT_SECTIONS = [
        r"experimental",
        r"materials?\s+and\s+methods?",
        r"results?",
        r"discussion",
        r"characterization",
        r"electrochemical",
        r"synthesis",
        r"preparation"
    ]
    
    # Sections to exclude
    EXCLUDE_SECTIONS = [
        r"references?",
        r"bibliography",
        r"acknowledgments?",
        r"supplementary",
        r"supporting\s+information"
    ]
    
    def __init__(self):
        self.chunk_size = config.chunk_size
    
    @staticmethod
    def extract_text(pdf_path: Path) -> str:
        """Extract raw text from PDF preserving layout."""
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []
            
            for page in doc:
                # Extract text with layout preservation
                text = page.get_text("text")
                text_blocks.append(text)
            
            doc.close()
            return "\n\n".join(text_blocks)
        
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise
    
    def _remove_headers_footers(self, text: str) -> str:
        """Remove common headers, footers, and page numbers."""
        lines = text.split('\n')
        cleaned = []
        
        for line in lines:
            # Skip short lines that are likely headers/footers
            if len(line.strip()) < 10:
                continue
            
            # Skip lines that are just page numbers
            if re.match(r'^\s*\d+\s*$', line):
                continue
            
            # Skip lines with common header/footer patterns
            if re.search(r'(doi:|http://|https://|©\s*\d{4})', line, re.I):
                continue
            
            cleaned.append(line)
        
        return '\n'.join(cleaned)
    
    def _identify_section_boundaries(self, text: str) -> dict:
        """Identify start positions of different sections."""
        sections = {}
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check for relevant sections
            for pattern in self.RELEVANT_SECTIONS:
                if re.search(pattern, line_lower):
                    section_name = line.strip()
                    sections[section_name] = i
                    break
            
            # Check for sections to exclude
            for pattern in self.EXCLUDE_SECTIONS:
                if re.search(pattern, line_lower):
                    sections[f"EXCLUDE_{line.strip()}"] = i
                    break
        
        return sections
    
    def identify_relevant_sections(self, text: str) -> str:
        """
        Extract the full text for LLMs with large context windows (1m+ tokens).
        """
        # Remove headers/footers first
        text = self._remove_headers_footers(text)
        
        logger.info(
            f"Extracted {len(text)} chars (100.0%) for full-document analysis"
        )
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Normalize whitespace and remove artifacts."""
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove multiple newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove hyphenation at line breaks
        text = re.sub(r'-\n', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for processing.
        
        Args:
            text: Input text
            overlap: Number of characters to overlap between chunks
        
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end in last 100 chars
                chunk_text = text[start:end]
                last_period = chunk_text.rfind('. ')
                
                if last_period > self.chunk_size - 200:
                    end = start + last_period + 1
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
    
    def process(self, pdf_path: Path) -> dict:
        """
        Complete processing pipeline for a PDF.
        
        Returns:
            Dictionary with cleaned text and metadata
        """
        logger.info(f"Processing {pdf_path.name}")
        
        # Extract raw text
        raw_text = self.extract_text(pdf_path)
        
        # Identify and extract relevant sections
        relevant_text = self.identify_relevant_sections(raw_text)
        
        # Clean whitespace
        cleaned_text = self._clean_whitespace(relevant_text)
        
        # Create chunks if needed
        chunks = self.chunk_text(cleaned_text)
        
        return {
            "filename": pdf_path.name,
            "raw_length": len(raw_text),
            "cleaned_length": len(cleaned_text),
            "num_chunks": len(chunks),
            "chunks": chunks,
            "full_text": cleaned_text
        }


def main():
    """CLI entry point for testing cleaner."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.cleaner <pdf_path>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    cleaner = PDFCleaner()
    result = cleaner.process(pdf_path)
    
    print(f"\nProcessed: {result['filename']}")
    print(f"Raw length: {result['raw_length']:,} chars")
    print(f"Cleaned length: {result['cleaned_length']:,} chars")
    print(f"Chunks: {result['num_chunks']}")
    print(f"\nFirst 500 chars:\n{result['full_text'][:500]}")


if __name__ == "__main__":
    main()
