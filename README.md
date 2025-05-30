# easycourse
Vector search system for PDF documents that can handle both regular PDFs and scanned documents with OCR. Built with FastAPI, FAISS, and sentence-transformers.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR (required for processing scanned PDFs):
- On macOS: `brew install tesseract`
- On Ubuntu: `sudo apt-get install tesseract-ocr`
- On Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

3. Install poppler (required for PDF to image conversion):
- On macOS: `brew install poppler`
- On Ubuntu: `sudo apt-get install poppler-utils`
- On Windows: Download and install from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases/)

## Running the Server

Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload
```

The server will be available at `http://localhost:8000`

## API Endpoints

### Upload PDF
- **POST** `/upload`
- Upload a PDF file to be indexed
- Content-Type: multipart/form-data
- File field name: `file`

### Search
- **GET** `/search?query=your search query&k=5`
- Search the indexed PDFs
- Parameters:
  - `query`: Search query string
  - `k`: Number of results to return (default: 5)

### Status
- **GET** `/status`
- Get the current status of the search index

### Summarize PDF
- **POST** `/summary`
- Upload a PDF and receive a short summary

### Translate Text
- **POST** `/translate`
- Translate text from English to another language using JSON body

## Example Usage

1. Upload a PDF:
```bash
curl -X POST -F "file=@your_document.pdf" http://localhost:8000/upload
```

2. Search the indexed PDFs:
```bash
curl "http://localhost:8000/search?query=your search query"
```

3. Summarize a PDF:
```bash
curl -X POST -F "file=@your_document.pdf" http://localhost:8000/summary
```

4. Translate text to Spanish:
```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"text": "Hello world", "target_lang": "es"}' \
    http://localhost:8000/translate
```

5. Ask a question about the indexed PDFs:
```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"question": "What is the main topic?"}' \
    http://localhost:8000/answer
```

## Features

- Handles both regular PDFs and scanned documents with OCR
- Uses sentence-transformers for semantic search
- FAISS for efficient vector similarity search
- FastAPI for a modern, async API
- Stores index and metadata for persistence
- Generate summaries of uploaded PDFs
- Translate text results into other languages
- Automatically loads an existing index on startup
- Keeps track of page numbers for more accurate results
- Answer questions about PDFs using retrieval-augmented QA
