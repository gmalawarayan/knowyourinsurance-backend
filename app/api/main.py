"""
Insurance Policy Analyzer API

FastAPI application for the insurance policy analysis system.
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import shutil
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Policy Analyzer API",
    description="API for analyzing insurance policy documents",
    version="1.0.0"
)

# Import your auth module
from app.core.auth import (
    User, authenticate_user, create_access_token, 
    get_current_active_user, has_role, ACCESS_TOKEN_EXPIRE_MINUTES
)

# Add these models
class Token(BaseModel):
    access_token: str
    token_type: str

class UserInDB(BaseModel):
    username: str
    email: str
    role: str

# Add login endpoint
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    from app.core.auth import fake_users_db  # Import here to avoid circular imports
    
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserInDB)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return {
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role
    }

# Adjust import paths based on your macOS setup if needed
# If you used the PyPDF2 modification:
# from app.processors.document_processor_macos import DocumentProcessor
# else:
from app.processors import DocumentProcessor, PolicyChunker, InsuranceEntityExtractor

from app.core import VectorStore, LLMQuerySystem, InsuranceAnalyzer

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create upload directory relative to where the script is run
# If running from root main.py, this path might need adjustment or absolute paths
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize components
# Ensure persist_directory points correctly relative to the root execution
vector_store = VectorStore(persist_directory="./chroma_db")
llm_query_system = LLMQuerySystem(vector_store=vector_store)
insurance_analyzer = InsuranceAnalyzer(llm_query_system=llm_query_system, vector_store=vector_store)

# Use the correct DocumentProcessor based on your setup
# If you used the PyPDF2 modification:
# document_processor = DocumentProcessor()
# else:
document_processor = DocumentProcessor()

policy_chunker = PolicyChunker()
entity_extractor = InsuranceEntityExtractor()

# In-memory storage for processed documents (Consider a more persistent solution for production)
documents = {}

# Pydantic models for API
class QueryRequest(BaseModel):
    question: str
    document_id: str
    query_type: Optional[str] = "general"

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class DocumentResponse(BaseModel):
    document_id: str
    is_insurance_policy: bool
    metadata: Dict[str, Any]
    entities: Dict[str, Any]
    summary: Optional[str] = None

class AnalysisResponse(BaseModel):
    analysis: Dict[str, Any]

@app.get("/")
async def root():
    """Root endpoint to check if API is running."""
    return {"message": "Insurance Policy Analyzer API is running"}

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    # current_user: User = Depends(has_role(["admin", "analyst", "user"]))  # All roles can upload
):
    # Add user info to the log
    logger.info(f"User uploaded document: {file.filename}")
    """
    Upload and process an insurance policy document.

    Args:
        file: PDF file to upload

    Returns:
        Document ID and extracted metadata
    """
    logger.info(f"Received document upload: {file.filename}")

    # Generate unique document ID
    document_id = str(uuid.uuid4())

    # Save file to disk
    file_path = UPLOAD_DIR / f"{document_id}_{file.filename}"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    # Process document
    try:
        # Process document
        processed_doc = document_processor.process_document(str(file_path))

        # Store document data
        documents[document_id] = {
            "file_path": str(file_path),
            "filename": file.filename,
            "processed_data": processed_doc,
            "is_insurance_policy": processed_doc["is_insurance_policy"]
        }

        # If it's an insurance policy, process it further in the background
        if processed_doc["is_insurance_policy"]:
            background_tasks.add_task(
                process_insurance_policy,
                document_id,
                processed_doc
            )

            # Generate a quick summary
            summary_prompt = f"""
            Provide a brief summary (3-5 sentences) of the following insurance policy:

            {processed_doc['text'][:2000]}...

            Summary:
            """
            summary = llm_query_system.llm.invoke(summary_prompt)
        else:
            summary = "This document does not appear to be an insurance policy."

        return {
            "document_id": document_id,
            "is_insurance_policy": processed_doc["is_insurance_policy"],
            "metadata": processed_doc["metadata"],
            "entities": processed_doc["entities"],
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        # Clean up saved file if processing fails
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/documents/{document_id}/query", response_model=QueryResponse)
async def query_document(document_id: str, query_request: QueryRequest):
    """
    Query an insurance policy document.

    Args:
        document_id: Document ID
        query_request: Query request with question and query type

    Returns:
        Answer and source chunks
    """
    logger.info(f"Received query for document {document_id}: {query_request.question}")

    # Check if document exists and is processed
    if document_id not in documents or "processed_data" not in documents[document_id]:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found or not processed yet")

    # Check if document is an insurance policy
    if not documents[document_id]["is_insurance_policy"]:
        return {
            "answer": "This document does not appear to be an insurance policy. I can only answer questions about insurance policies.",
            "sources": []
        }

    # Detect query type if not specified
    query_type = query_request.query_type
    if not query_type or query_type == "auto":
        query_type = llm_query_system.detect_query_type(query_request.question)

    # Query document
    try:
        result = llm_query_system.query(
            question=query_request.question,
            document_id=document_id,
            query_type=query_type
        )

        return result

    except Exception as e:
        logger.error(f"Error querying document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")

@app.get("/documents/{document_id}/coverage-gaps", response_model=AnalysisResponse)
async def analyze_coverage_gaps(document_id: str):
    """
    Analyze coverage gaps in an insurance policy.

    Args:
        document_id: Document ID

    Returns:
        Coverage gap analysis
    """
    logger.info(f"Analyzing coverage gaps for document {document_id}")

    # Check if document exists and is processed
    if document_id not in documents or "processed_data" not in documents[document_id]:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found or not processed yet")

    # Check if document is an insurance policy
    if not documents[document_id]["is_insurance_policy"]:
        return {
            "analysis": {
                "summary": "This document does not appear to be an insurance policy.",
                "gaps": []
            }
        }

    # Analyze coverage gaps
    try:
        result = insurance_analyzer.analyze_coverage_gaps(document_id)

        return {"analysis": result}

    except Exception as e:
        logger.error(f"Error analyzing coverage gaps: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing coverage gaps: {str(e)}")

@app.get("/documents/{document_id}/risk-assessment", response_model=AnalysisResponse)
async def assess_risk(document_id: str):
    """
    Assess risk based on an insurance policy.

    Args:
        document_id: Document ID

    Returns:
        Risk assessment
    """
    logger.info(f"Assessing risk for document {document_id}")

    # Check if document exists and is processed
    if document_id not in documents or "processed_data" not in documents[document_id]:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found or not processed yet")

    # Check if document is an insurance policy
    if not documents[document_id]["is_insurance_policy"]:
        return {
            "analysis": {
                "overall_risk_level": "Unknown",
                "summary": "This document does not appear to be an insurance policy.",
                "risk_factors": []
            }
        }

    # Assess risk
    try:
        result = insurance_analyzer.assess_risk(document_id)

        return {"analysis": result}

    except Exception as e:
        logger.error(f"Error assessing risk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error assessing risk: {str(e)}")

@app.get("/documents/{document_id}/premium-analysis", response_model=AnalysisResponse)
async def analyze_premium(document_id: str):
    """
    Analyze premium structure of an insurance policy.

    Args:
        document_id: Document ID

    Returns:
        Premium analysis
    """
    logger.info(f"Analyzing premium for document {document_id}")

    # Check if document exists and is processed
    if document_id not in documents or "processed_data" not in documents[document_id]:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found or not processed yet")

    # Check if document is an insurance policy
    if not documents[document_id]["is_insurance_policy"]:
        return {
            "analysis": {
                "total_premium": "Unknown",
                "payment_schedule": "Unknown",
                "summary": "This document does not appear to be an insurance policy.",
                "factors": []
            }
        }

    # Analyze premium
    try:
        result = insurance_analyzer.analyze_premium(document_id)

        return {"analysis": result}

    except Exception as e:
        logger.error(f"Error analyzing premium: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing premium: {str(e)}")

@app.get("/documents/{document_id}/benchmarks", response_model=AnalysisResponse)
async def identify_benchmarks(document_id: str, policy_type: Optional[str] = None):
    """
    Compare policy against industry benchmarks.

    Args:
        document_id: Document ID
        policy_type: Optional policy type override

    Returns:
        Benchmark comparison
    """
    logger.info(f"Identifying benchmarks for document {document_id}")

    # Check if document exists and is processed
    if document_id not in documents or "processed_data" not in documents[document_id]:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found or not processed yet")

    # Check if document is an insurance policy
    if not documents[document_id]["is_insurance_policy"]:
        return {
            "analysis": {
                "policy_type": policy_type or "Unknown",
                "summary": "This document does not appear to be an insurance policy.",
                "benchmarks": []
            }
        }

    # Identify benchmarks
    try:
        result = insurance_analyzer.identify_policy_benchmarks(document_id, policy_type)

        return {"analysis": result}

    except Exception as e:
        logger.error(f"Error identifying benchmarks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error identifying benchmarks: {str(e)}")

@app.post("/documents/compare", response_model=AnalysisResponse)
async def compare_policies(document_id_1: str = Form(...), document_id_2: str = Form(...)):
    """
    Compare two insurance policies.

    Args:
        document_id_1: Document ID of the first policy
        document_id_2: Document ID of the second policy

    Returns:
        Policy comparison
    """
    logger.info(f"Comparing policies {document_id_1} and {document_id_2}")

    # Check if documents exist and are processed
    if document_id_1 not in documents or "processed_data" not in documents[document_id_1]:
        raise HTTPException(status_code=404, detail=f"Document {document_id_1} not found or not processed yet")

    if document_id_2 not in documents or "processed_data" not in documents[document_id_2]:
        raise HTTPException(status_code=404, detail=f"Document {document_id_2} not found or not processed yet")

    # Check if documents are insurance policies
    if not documents[document_id_1]["is_insurance_policy"]:
        return {
            "analysis": {
                "summary": f"Document {document_id_1} does not appear to be an insurance policy.",
                "differences": []
            }
        }

    if not documents[document_id_2]["is_insurance_policy"]:
        return {
            "analysis": {
                "summary": f"Document {document_id_2} does not appear to be an insurance policy.",
                "differences": []
            }
        }

    # Compare policies
    try:
        result = insurance_analyzer.compare_policies(document_id_1, document_id_2)

        return {"analysis": result}

    except Exception as e:
        logger.error(f"Error comparing policies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing policies: {str(e)}")

@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents():
    """
    List all uploaded documents.

    Returns:
        List of documents with metadata
    """
    logger.info("Listing all documents")

    result = []

    for doc_id, doc_data in documents.items():
        # Ensure processed_data exists before accessing it
        if "processed_data" in doc_data:
            processed_data = doc_data["processed_data"]
            result.append({
                "document_id": doc_id,
                "is_insurance_policy": processed_data["is_insurance_policy"],
                "metadata": processed_data["metadata"],
                "entities": processed_data["entities"],
                "summary": None  # Don't include summary in list to keep response size small
            })
        else:
            # Handle case where document might be uploaded but not yet processed
            result.append({
                "document_id": doc_id,
                "is_insurance_policy": False,
                "metadata": {"filename": doc_data.get("filename", "Unknown")},
                "entities": {},
                "summary": "Processing..."
            })

    return result

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document.

    Args:
        document_id: Document ID

    Returns:
        Success message
    """
    logger.info(f"Deleting document {document_id}")

    # Check if document exists
    if document_id not in documents:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    # Get file path
    file_path = Path(documents[document_id]["file_path"])

    # Delete file if it exists
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            logger.warning(f"Error deleting file {file_path}: {str(e)}")

    # Delete from vector store
    try:
        vector_store.delete_document(document_id)
    except Exception as e:
        logger.warning(f"Error deleting document from vector store: {str(e)}")

    # Delete from in-memory storage
    del documents[document_id]

    return {"message": f"Document {document_id} deleted successfully"}

async def process_insurance_policy(document_id: str, processed_doc: Dict[str, Any]):
    """
    Process an insurance policy document in the background.

    Args:
        document_id: Document ID
        processed_doc: Processed document data
    """
    logger.info(f"Processing insurance policy {document_id} in background")

    try:
        # Extract entities
        entities = entity_extractor.extract_entities(processed_doc["text"])

        # Update document data with extracted entities
        if document_id in documents and "processed_data" in documents[document_id]:
            documents[document_id]["processed_data"]["entities"].update(entities)
        else:
            logger.warning(f"Document {document_id} not found in memory for entity update.")
            return # Exit if document disappeared

        # Chunk document
        chunks = policy_chunker.chunk_document(
            processed_doc["text"],
            {"document_id": document_id}
        )

        # Add document to vector store
        vector_store.add_document(
            document_id=document_id,
            chunks=chunks,
            metadata={
                "filename": documents[document_id]["filename"],
                "document_id": document_id,
                "policy_type": entities.get("policy_type", "Unknown"),
                **processed_doc["metadata"]
            }
        )

        logger.info(f"Successfully processed insurance policy {document_id}")

    except Exception as e:
        logger.error(f"Error processing insurance policy {document_id}: {str(e)}")

# This block allows running the API directly for development/testing
# but the primary way to run is via the root main.py
if __name__ == "__main__":
    # Note: Relative paths for UPLOAD_DIR and chroma_db might behave differently
    # when running this file directly compared to running from the root main.py
    logger.warning("Running API directly from app/api/main.py. Use root main.py for standard execution.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

