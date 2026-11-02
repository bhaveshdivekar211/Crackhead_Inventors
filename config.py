"""Shared configuration for ingest.py and agent.py"""

from pathlib import Path
import pyaudio

# ===============
# FILE PATHS
# ===============
DOCS_DIR = Path("docs")
FAISS_INDEX_DIR = Path("faiss_index")
MANIFEST_PATH = Path("manifest.json")

# ===============
# EMBEDDING & RETRIEVAL
# ===============
EMBEDDING_MODEL = "bge-m3"
RETRIEVAL_K = 3
# ===============
# CHUNKING SETTINGS
# ===============
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 16

# ===============
# AUDIO CONFIGURATION
# ===============
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000
AUDIO_CHUNK_SIZE = 512
AUDIO_SILENCE_THRESHOLD = 2.0
# ===============
# GEMINI LIVE SETTINGS
# ===============
GEMINI_MODEL = "gemini-3.8-live"

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant who answers questions accurately "
    "by referring to the provided documents using the search_docs tool. "
    "After you search the documents and get results, ALWAYS synthesize and speak "
    "a clear, natural response using the information found. "
    "Speak in a conversational, natural human tone. If the answer isn't "
    "in the documents, say so instead of guessing. "
    "Your responses must be spoken aloud - always provide audio output."
)

# ===============
# LOGGING
# ===============
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
