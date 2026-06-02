"""Configuration for the MXene data pipeline."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Pipeline configuration with validation."""
    
    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    pdfs_downloaded: Path = Field(default_factory=lambda: Path("data/pdfs_downloaded"))
    pdfs_manual: Path = Field(default_factory=lambda: Path("data/pdfs_manual"))
    processed_json: Path = Field(default_factory=lambda: Path("data/processed_json"))
    logs_dir: Path = Field(default_factory=lambda: Path("data/logs"))
    
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3")
    ollama_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    ollama_timeout: int = Field(default=120)
    
    # Semantic Scholar API
    s2_api_key: Optional[str] = Field(default=None)
    s2_rate_limit_delay: tuple[float, float] = Field(default=(2.0, 5.0))
    s2_max_results: int = Field(default=100)
    
    # Retry Configuration
    max_retries: int = Field(default=3)
    retry_wait_min: int = Field(default=2)
    retry_wait_max: int = Field(default=10)
    
    # Processing (Increased for full-document extraction to preserve context)
    chunk_size: int = Field(default=50000)
    max_context_length: int = Field(default=100000)
    
    # Search Query
    search_query: str = Field(default="MXene supercapacitor electrochemical performance")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Make paths absolute
        self.pdfs_downloaded = self.base_dir / self.pdfs_downloaded
        self.pdfs_manual = self.base_dir / self.pdfs_manual
        self.processed_json = self.base_dir / self.processed_json
        self.logs_dir = self.base_dir / self.logs_dir
        
        # Create directories
        for path in [self.pdfs_downloaded, self.pdfs_manual, 
                     self.processed_json, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load configuration from environment variables."""
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
            s2_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        )


# Global config instance
config = PipelineConfig.from_env()
