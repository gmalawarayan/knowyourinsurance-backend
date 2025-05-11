"""
Main application entry point for the Insurance Policy Analyzer
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("./uploads", exist_ok=True)
os.makedirs("./chroma_db", exist_ok=True)

# Import and run the FastAPI app
from app.api.main import app

if __name__ == "__main__":
    import uvicorn

is_prod = os.environ.get("ENVIRONMENT", "development") == "production"
    
if is_prod:
    # Production SSL paths (Let's Encrypt)
    ssl_keyfile = "/etc/letsencrypt/live/yourdomain.com/privkey.pem"
    ssl_certfile = "/etc/letsencrypt/live/yourdomain.com/fullchain.pem"
else:
    # Development SSL paths (self-signed)
    ssl_keyfile = "./certs/key.pem"
    ssl_certfile = "./certs/cert.pem"

logger.info("Starting Insurance Policy Analyzer API with SSL/TLS")
uvicorn.run(
    "app.api.main:app", 
    host="127.0.0.1", 
    port=8000, 
    reload=False,
    ssl_keyfile=ssl_keyfile,
    ssl_certfile=ssl_certfile
)