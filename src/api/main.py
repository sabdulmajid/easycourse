from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
import os
import shutil
from pathlib import Path

from ..summarizer.summary import SummaryGenerator
from ..translator.translator import TextTranslator


from ..pdf_processor.processor import PDFProcessor
from ..embeddings.generator import EmbeddingGenerator
from ..vector_search.index import VectorSearch

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pdf_processor = PDFProcessor()
embedding_generator = EmbeddingGenerator()
vector_search = None
summary_generator = SummaryGenerator()
translator_cache = {}

# Create directories for uploads and index
UPLOAD_DIR = Path("uploads")
INDEX_DIR = Path("index")
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# Attempt to load an existing index on startup
if (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "data.pkl").exists():
    try:
        vector_search = VectorSearch.load(str(INDEX_DIR))
    except Exception:
        vector_search = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file and add it to the search index."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save the uploaded file
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the PDF
    chunks, page_numbers = pdf_processor.process_pdf(str(file_path))
    
    if not chunks:
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
    
    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(chunks)
    
    # Prepare metadata
    metadata = [{"source": file.filename, "page": page} for page in page_numbers]
    
    # Initialize or load vector search
    global vector_search
    if vector_search is None:
        vector_search = VectorSearch(embedding_generator.get_dimension())
    
    # Add to index
    vector_search.add_vectors(embeddings, chunks, metadata)
    
    # Save the index
    vector_search.save(str(INDEX_DIR))

    return {"message": f"Successfully processed {len(chunks)} chunks from {file.filename}"}


@app.post("/summary")
async def summarize_pdf(file: UploadFile = File(...)):
    """Upload a PDF file and return a short summary."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks, _ = pdf_processor.process_pdf(str(file_path))
    if not chunks:
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

    text = " ".join(chunks[:50])
    summary = summary_generator.summarize(text)
    return {"summary": summary}


class TranslationRequest(BaseModel):
    text: str
    target_lang: str = "es"


@app.post("/translate")
async def translate_text(req: TranslationRequest):
    """Translate text from English to a target language."""
    if req.target_lang not in translator_cache:
        translator_cache[req.target_lang] = TextTranslator(req.target_lang)

    translator = translator_cache[req.target_lang]
    translation = translator.translate(req.text)
    return {"translation": translation}

@app.get("/search")
async def search(query: str, k: int = 5):
    """Search the indexed PDFs for relevant content."""
    if vector_search is None:
        raise HTTPException(status_code=400, detail="No PDFs have been indexed yet")
    
    # Generate embedding for query
    query_embedding = embedding_generator.generate_embedding(query)
    
    # Search
    results = vector_search.search(query_embedding, k)
    
    # Format results
    formatted_results = []
    for text, metadata, distance in results:
        # Split text into lines and clean up
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Find the most relevant line
        best_line = None
        best_score = 0
        query_words = set(query.lower().split())
        
        for line in lines:
            # Calculate semantic similarity score (from FAISS)
            # FAISS returns cosine similarity for normalized embeddings
            semantic_score = (distance + 1) / 2
            
            # Calculate exact match score
            line_words = set(line.lower().split())
            exact_matches = len(query_words.intersection(line_words))
            exact_score = min(1.0, exact_matches / len(query_words))
            
            # Combine scores with weights
            # 70% semantic similarity, 30% exact matches
            combined_score = (0.7 * semantic_score) + (0.3 * exact_score)
            
            if combined_score > best_score:
                best_score = combined_score
                best_line = line
        
        # If no line matches, use the first non-empty line
        if not best_line and lines:
            best_line = lines[0]
            best_score = 0.1  # Low confidence for fallback
        
        # Format score as percentage
        confidence = round(best_score * 100, 1)
        
        formatted_results.append({
            "text": best_line if best_line else text,
            "source": metadata["source"],
            "page": metadata["page"],
            "confidence": f"{confidence}%"
        })
    
    # Sort results by confidence
    formatted_results.sort(key=lambda x: float(x["confidence"].strip("%")), reverse=True)
    
    return {"results": formatted_results}

@app.get("/status")
async def status():
    """Get the current status of the search index."""
    if vector_search is None:
        return {"status": "No PDFs indexed"}
    
    return {
        "status": "Ready",
        "num_documents": len(vector_search.texts),
        "dimension": vector_search.dimension
    }
