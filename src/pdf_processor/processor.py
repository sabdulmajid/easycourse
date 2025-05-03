import os
from typing import List, Tuple
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
import numpy as np
from PIL import Image
import nltk
from nltk.tokenize import sent_tokenize

class PDFProcessor:
    def __init__(self, min_text_length: int = 50):
        """
        Initialize the PDF processor.
        
        Args:
            min_text_length: Minimum length of text to consider as a valid chunk
        """
        self.min_text_length = min_text_length
        # Download the NLTK data for sentence tokenization
        nltk.download('punkt_tab')

    def _extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract text from a regular PDF file."""
        chunks = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Split text into sentences
                sentences = sent_tokenize(text)
                chunks.extend(sentences)
        
        return chunks

    def _extract_text_from_scanned_pdf(self, pdf_path: str) -> List[str]:
        """Extract text from a scanned PDF using OCR."""
        chunks = []
        
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        
        for image in images:
            # Convert image to grayscale for better OCR
            if image.mode != 'L':
                image = image.convert('L')
            
            # Perform OCR
            text = pytesseract.image_to_string(image)
            
            # Split text into paragraphs and filter out empty ones
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            chunks.extend(paragraphs)
        
        return chunks

    def process_pdf(self, pdf_path: str) -> Tuple[List[str], List[int]]:
        """
        Process a PDF file and return text chunks with their page numbers.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (chunks, page_numbers) where chunks are the text segments
            and page_numbers are the corresponding page numbers
        """
        # First try regular text extraction
        chunks = self._extract_text_from_pdf(pdf_path)
        
        # If we got very little text, assume it's a scanned document
        if len(' '.join(chunks)) < 100:
            chunks = self._extract_text_from_scanned_pdf(pdf_path)
        
        # Filter out very short chunks
        filtered_chunks = []
        page_numbers = []
        current_page = 0
        
        for chunk in chunks:
            if len(chunk) >= self.min_text_length:
                filtered_chunks.append(chunk)
                page_numbers.append(current_page)
            current_page += 1
        
        return filtered_chunks, page_numbers

    def process_pdf_directory(self, directory_path: str) -> Tuple[List[str], List[str], List[int]]:
        """
        Process all PDFs in a directory.
        
        Args:
            directory_path: Path to the directory containing PDF files
            
        Returns:
            Tuple of (chunks, source_files, page_numbers)
        """
        all_chunks = []
        source_files = []
        page_numbers = []
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                chunks, pages = self.process_pdf(file_path)
                
                all_chunks.extend(chunks)
                source_files.extend([filename] * len(chunks))
                page_numbers.extend(pages)
        
        return all_chunks, source_files, page_numbers