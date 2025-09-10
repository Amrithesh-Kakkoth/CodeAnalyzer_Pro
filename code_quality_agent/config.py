"""
Configuration management for the Code Quality Intelligence Agent.
"""

import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AnalysisConfig(BaseModel):
    """Configuration for code analysis."""
    
    max_file_size_mb: int = Field(default=10, description="Maximum file size to analyze in MB")
    supported_languages: List[str] = Field(
        default=["python", "javascript", "typescript"],
        description="Supported programming languages"
    )
    default_severity_threshold: str = Field(
        default="medium",
        description="Default severity threshold for reporting issues"
    )
    enable_security_scan: bool = Field(default=True, description="Enable security vulnerability scanning")
    enable_duplication_detection: bool = Field(default=True, description="Enable code duplication detection")
    enable_complexity_analysis: bool = Field(default=True, description="Enable complexity analysis")
    enable_testing_analysis: bool = Field(default=True, description="Enable testing gap analysis")
    enable_documentation_analysis: bool = Field(default=True, description="Enable documentation analysis")


class RAGConfig(BaseModel):
    """Configuration for RAG (Retrieval-Augmented Generation) system."""
    
    vector_db_path: str = Field(default="./data/vector_db", description="Path to vector database")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for vectorization"
    )
    chunk_size: int = Field(default=1000, description="Text chunk size for vectorization")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")


class GitHubConfig(BaseModel):
    """Configuration for GitHub API integration."""
    
    api_token: str = Field(default="", description="GitHub API token for higher rate limits")
    base_url: str = Field(default="https://api.github.com", description="GitHub API base URL")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")


class AIConfig(BaseModel):
    """Configuration for AI models and services."""
    
    # LLM Provider Selection
    llm_provider: str = Field(default="groq", description="LLM provider: groq, openai")
    
    # Groq Configuration (Free tier available)
    groq_api_key: str = Field(default="", description="Groq API key (free tier)")
    groq_model_name: str = Field(default="llama-3.1-8b-instant", description="Groq model name")
    
    # OpenAI Configuration (Paid - kept for fallback)
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model_name: str = Field(default="gpt-3.5-turbo", description="OpenAI model to use")
    
    # General LLM Settings
    temperature: float = Field(default=0.1, description="Temperature for AI responses")
    max_tokens: int = Field(default=2000, description="Maximum tokens for AI responses")




class Config(BaseModel):
    """Main configuration class."""
    
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    ai: AIConfig = Field(
        default_factory=lambda: AIConfig(
            llm_provider=os.getenv("LLM_PROVIDER", "groq"),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", "")
        )
    )
    github: GitHubConfig = Field(
        default_factory=lambda: GitHubConfig(
            api_token=os.getenv("GITHUB_API_TOKEN", "")
        )
    )
    
    class Config:
        env_prefix = "CQA_"


# Global configuration instance
config = Config()

# Update from environment variables
config.analysis.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
config.analysis.supported_languages = os.getenv("SUPPORTED_LANGUAGES", "python,javascript,typescript").split(",")
config.analysis.default_severity_threshold = os.getenv("DEFAULT_SEVERITY_THRESHOLD", "medium")

config.rag.vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
config.rag.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

