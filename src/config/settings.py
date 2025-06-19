"""
Configuration settings for the Second Brain AI Assistant.
This module manages all environment variables and application settings.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Main settings class for the Second Brain AI Assistant."""
    
    # Project settings
    PROJECT_NAME: str = "Second Brain AI Assistant"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    
    # API settings
    API_V1_STR: str = "/api/v1"
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Database settings (MongoDB)
    MONGODB_URL: str = Field(default="mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority", env="MONGODB_URL")
    DATABASE_NAME: str = Field(default="second_brain", env="DATABASE_NAME")
    COLLECTION_NAME: str = Field(default="knowledge_base", env="COLLECTION_NAME")
    
    # MongoDB Atlas Vector Search settings
    VECTOR_INDEX_NAME: str = Field(default="vector_index", env="VECTOR_INDEX_NAME")
    TEXT_INDEX_NAME: str = Field(default="text_index", env="TEXT_INDEX_NAME")
    VECTOR_DIMENSIONS: int = Field(default=384, env="VECTOR_DIMENSIONS")  # for all-MiniLM-L6-v2
    
    # Vector Database settings
    VECTOR_DB_PATH: str = Field(default="./data/chroma_db", env="VECTOR_DB_PATH")
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    CHUNK_SIZE: int = Field(default=1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # LLM settings
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    HUGGINGFACE_API_TOKEN: Optional[str] = Field(default=None, env="HUGGINGFACE_API_TOKEN")
    LLM_MODEL: str = Field(default="gpt-4o", env="LLM_MODEL")
    MAX_TOKENS: int = Field(default=4096, env="MAX_TOKENS")
    TEMPERATURE: float = Field(default=0.1, env="TEMPERATURE")
    
    # Notion API settings (for data collection) - Following DecodingML naming
    NOTION_SECRET_KEY: Optional[str] = Field(default=None, env="NOTION_SECRET_KEY")
    NOTION_API_KEY: Optional[str] = Field(default=None, env="NOTION_API_KEY")  # Fallback
    NOTION_DATABASE_ID: Optional[str] = Field(default=None, env="NOTION_DATABASE_ID")
    
    # Crawling settings
    MAX_CRAWL_PAGES: int = Field(default=100, env="MAX_CRAWL_PAGES")
    CRAWL_DELAY: float = Field(default=1.0, env="CRAWL_DELAY")
    USER_AGENT: str = Field(
        default="SecondBrainBot/1.0 (+https://github.com/your-username/second-brain)",
        env="USER_AGENT"
    )
    
    # MLOps settings
    COMET_API_KEY: Optional[str] = Field(default=None, env="COMET_API_KEY")
    COMET_PROJECT_NAME: str = Field(default="second-brain-ai", env="COMET_PROJECT_NAME")
    ZENML_STORE_URL: Optional[str] = Field(default=None, env="ZENML_STORE_URL")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        settings.DATA_DIR,
        settings.DATA_DIR / "raw",
        settings.DATA_DIR / "processed", 
        settings.DATA_DIR / "embeddings",
        settings.LOGS_DIR,
        settings.MODELS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test settings loading
    print(f"Project: {settings.PROJECT_NAME}")
    print(f"Version: {settings.VERSION}")
    print(f"Debug: {settings.DEBUG}")
    ensure_directories()
    print("✅ Settings loaded and directories created successfully!") 