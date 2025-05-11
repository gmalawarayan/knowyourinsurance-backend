# Insurance Policy Analyzer - Implementation and Deployment Guide

This guide provides instructions for setting up, running, and deploying the Insurance Policy Analyzer backend.

## Overview

The Insurance Policy Analyzer is an AI-powered system that allows users to:
- Upload insurance policy documents (PDF)
- Extract key information from policies
- Ask questions about policies in natural language
- Analyze coverage gaps and risks
- Compare policies against benchmarks or other policies
- Assess premium structures and risk profiles

## System Requirements

- Python 3.10+
- 8GB+ RAM (16GB recommended for optimal performance)
- 4GB+ free disk space
- Internet connection (if using OpenAI models)
- CUDA-compatible GPU (optional, for faster local model inference)

## Installation

1. Clone the repository or extract the source code:
   ```
   git clone <repository-url>
   cd insurance-policy-analyzer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional, for OpenAI integration):
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

## Running the Application

1. Start the API server:
   ```
   python main.py
   ```

2. The API will be available at `http://localhost:8000`

3. Access the API documentation at `http://localhost:8000/docs`

## API Endpoints

The system provides the following API endpoints:

- `POST /documents/upload` - Upload and process an insurance policy document
- `POST /documents/{document_id}/query` - Query an insurance policy document
- `GET /documents/{document_id}/coverage-gaps` - Analyze coverage gaps
- `GET /documents/{document_id}/risk-assessment` - Assess risk based on policy
- `GET /documents/{document_id}/premium-analysis` - Analyze premium structure
- `GET /documents/{document_id}/benchmarks` - Compare against industry benchmarks
- `POST /documents/compare` - Compare two insurance policies
- `GET /documents` - List all uploaded documents
- `DELETE /documents/{document_id}` - Delete a document

## Configuration Options

The system can be configured by modifying the following files:

- `app/core/llm_query.py` - LLM model settings (model type, temperature)
- `app/core/vector_store.py` - Vector database settings (embedding model, persistence)
- `app/processors/document_processor.py` - Document processing settings (OCR, chunking)

## Using Local LLMs with Ollama

By default, the system is configured to use Ollama for local LLM inference. To use this:

1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)

2. Pull the Llama 3 model:
   ```
   ollama pull llama3
   ```

3. Start the Ollama service:
   ```
   ollama serve
   ```

4. The system will automatically connect to the local Ollama instance.

## Using OpenAI Models

To use OpenAI models instead of local models:

1. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

2. Modify the LLMQuerySystem initialization in `app/api/main.py`:
   ```python
   llm_query_system = LLMQuerySystem(
       vector_store=vector_store,
       model_type="openai",
       model_name="gpt-4o"
   )
   ```

## Deployment Options

### Docker Deployment

1. Build the Docker image:
   ```
   docker build -t insurance-policy-analyzer .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 -v ./data:/app/data insurance-policy-analyzer
   ```

### Cloud Deployment

#### Railway

1. Create a new project on Railway
2. Connect your GitHub repository
3. Add the following environment variables:
   - `PORT=8000`
   - `OPENAI_API_KEY=your_api_key_here` (if using OpenAI)
4. Deploy the application

#### Digital Ocean App Platform

1. Create a new app on Digital Ocean App Platform
2. Connect your GitHub repository
3. Configure the app:
   - Type: Web Service
   - Build Command: `pip install -r requirements.txt`
   - Run Command: `python main.py`
   - HTTP Port: 8000
4. Add environment variables (if using OpenAI)
5. Deploy the application

## Integration with Frontend Applications

The API can be integrated with any frontend application using standard HTTP requests. Example integration with a React application:

```javascript
// Upload a policy document
const uploadPolicy = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8000/documents/upload', {
    method: 'POST',
    body: formData,
  });
  
  return await response.json();
};

// Query a policy document
const queryPolicy = async (documentId, question) => {
  const response = await fetch(`http://localhost:8000/documents/${documentId}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      document_id: documentId,
    }),
  });
  
  return await response.json();
};
```

## Customization and Extension

### Adding New Analysis Features

To add new analysis features:

1. Add new methods to the `InsuranceAnalyzer` class in `app/core/insurance_analyzer.py`
2. Add new API endpoints in `app/api/main.py`

### Supporting Additional Document Types

To support additional document types:

1. Modify the `_extract_from_pdf` method in `app/processors/document_processor.py`
2. Add new extraction methods for other file types

### Improving Entity Extraction

To improve entity extraction:

1. Enhance the regex patterns in `app/processors/entity_extractor.py`
2. Consider adding a named entity recognition model for better extraction

## Troubleshooting

### Common Issues

1. **Document processing fails**
   - Check if the PDF is text-based or scanned
   - Enable OCR for scanned documents
   - Check file permissions

2. **Vector database errors**
   - Delete the `chroma_db` directory and restart the application
   - Check disk space

3. **LLM connection issues**
   - Verify Ollama is running (for local models)
   - Check API key (for OpenAI models)
   - Check internet connection

4. **Memory issues**
   - Reduce chunk size in `app/processors/chunking.py`
   - Use a smaller embedding model
   - Process fewer documents simultaneously

## Performance Optimization

For better performance:

1. Use a GPU for faster embedding generation and LLM inference
2. Increase chunk size for better context understanding
3. Use a more powerful embedding model for better retrieval
4. Implement caching for frequent queries
5. Use a production ASGI server like Gunicorn with Uvicorn workers

## Security Considerations

1. Implement proper authentication and authorization
2. Encrypt sensitive data
3. Implement rate limiting
4. Use HTTPS in production
5. Regularly update dependencies

## Support and Maintenance

For support and maintenance:

1. Check logs in `app.log`
2. Monitor system resources
3. Regularly backup the `chroma_db` directory
4. Update dependencies regularly
