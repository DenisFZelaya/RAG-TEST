import os
from dotenv import load_dotenv

load_dotenv() # Carga variables de .env si existe

# --- Modelos ---
# Asegúrate de haber descargado estos modelos en Ollama:
# ollama pull llama3:8b
# ollama pull nomic-embed-text
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:8b") # Modelo de lenguaje a usar via Ollama
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5") # Modelo de embeddings local (HuggingFace)
# Alternativa ligera de embeddings: "all-MiniLM-L6-v2"

# --- Base de Datos Vectorial ---
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store") # Directorio para ChromaDB
COLLECTION_NAME = "project_guidelines" # Nombre de la colección en ChromaDB

# --- Procesamiento de Documentos ---
DOCUMENT_SOURCE_DIR = os.getenv("DOCUMENT_SOURCE_DIR", "./data") # Directorio de documentos fuente
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000)) # Tamaño de los fragmentos de texto
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150)) # Superposición entre fragmentos

# --- RAG ---
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 3)) # Número de fragmentos relevantes a recuperar

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434") # URL del servicio Ollama (nombre del servicio en docker-compose)