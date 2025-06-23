import os

class BaseConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY", os.urandom(24).hex())
    DEBUG = False
    TESTING = False
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_DIR = os.environ.get("LOG_DIR", "logs")
    
    DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", "oa_noncomm_xml_archives")
    EXTRACTED_DIR = os.path.join(DOWNLOAD_DIR, "extracted")
    PROCESSED_DIR = os.path.join(DOWNLOAD_DIR, "processed")
    FILTERED_DIR = os.path.join(DOWNLOAD_DIR, "filtered")

    HMDB_DATABASE_PATH = os.environ.get("HMDB_DATABASE_PATH", os.path.join("data", "hmdb_database.pkl"))

    REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
    REDIS_DB   = int(os.environ.get("REDIS_DB", 0))
    
    EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    EMBEDDINGS_DEVICE = os.environ.get("EMBEDDINGS_DEVICE", "cuda")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", os.path.join(BASE_DIR, "test_faiss_index"))
    
    CHUNK_THRESHOLD = int(os.environ.get("CHUNK_THRESHOLD", 2000))
    MAX_CHUNK_SIZE  = int(os.environ.get("MAX_CHUNK_SIZE", 2000))
    CHUNK_OVERLAP   = int(os.environ.get("CHUNK_OVERLAP", 150))
    
    BASE_URL = os.environ.get("BASE_URL", "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/")
    MAX_MEMORY_LIMIT_BYTES = int(os.environ.get("MAX_MEMORY_LIMIT_BYTES", 1 * 1024 * 1024 * 1024))
    AVERAGE_XML_SIZE_BYTES = int(os.environ.get("AVERAGE_XML_SIZE_BYTES", 100 * 1024))
    MAX_ARCHIVE_SIZE_MB = int(os.environ.get("MAX_ARCHIVE_SIZE_MB", 500))
    
    KEYWORDS = {
        "NNEC": [
            "necrotizing", "enterocolitis", "neonatal", "nnec",
            "neonatal necrotizing enterocolitis",
            "intestinal perforation", "gut ischemia", "inflammation",
            "prematurity", "premature infant", "low birth weight", "sepsis",
            "bacterial translocation", "circulatory failure"
        ],
        "biomarkers": [
            "c-reactive protein", "crp",
            "cytokine", "interleukin", "predictor",
            "calprotectin", "procalcitonin", "soluble cd14",
            "glyco-redox", "gut microbiome", "fucose", "microbial metabolites"
        ]
    }

    CELERY_BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    CELERY_RESULT_BACKEND = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

class ProductionConfig(BaseConfig):
    DEBUG = False
    TESTING = False

class TestingConfig(BaseConfig):
    DEBUG = True
    TESTING = True

config = {
    "production": ProductionConfig,
    "testing": TestingConfig,
}