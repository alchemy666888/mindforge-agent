"""
Application settings and configuration management.
Uses pydantic-settings for environment variable handling.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    deepseek_api_key: Optional[str] = Field(default=None, description="DeepSeek API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    serper_api_key: Optional[str] = Field(default=None, description="Serper search API key")
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily search API key")

    # Local LLM Settings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field(default="llama3.1:8b", description="Default Ollama model")

    # Application Settings
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    data_dir: Path = Field(default=Path("./data"), description="Data directory")
    output_dir: Path = Field(default=Path("./output"), description="Output directory")

    # Crawler Settings
    crawl_delay: float = Field(default=2.0, description="Delay between requests in seconds")
    max_articles: int = Field(default=100, description="Maximum articles to crawl")
    respect_robots_txt: bool = Field(default=True, description="Respect robots.txt")
    user_agent: str = Field(
        default="MindForge/1.0 (Research Bot; +https://github.com/mindforge)",
        description="User agent string for crawler",
    )

    # Generation Settings
    default_model: str = Field(default="deepseek/deepseek-chat", description="Default LLM model")
    fallback_model: str = Field(default="ollama/llama3.1:8b", description="Fallback LLM model")
    max_tokens: int = Field(default=4096, description="Maximum tokens for generation")
    temperature: float = Field(default=0.7, description="Generation temperature")

    # Dan Koe Specific
    dankoe_url: str = Field(
        default="https://letters.thedankoe.com/", description="Dan Koe newsletter URL"
    )

    def get_data_path(self, subdir: str = "") -> Path:
        """Get path within data directory."""
        path = self.data_dir / subdir if subdir else self.data_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_output_path(self, subdir: str = "") -> Path:
        """Get path within output directory."""
        path = self.output_dir / subdir if subdir else self.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_raw_data_path(self) -> Path:
        """Get raw data directory path."""
        return self.get_data_path("raw")

    def get_processed_data_path(self) -> Path:
        """Get processed data directory path."""
        return self.get_data_path("processed")

    def get_available_llm_model(self) -> str:
        """Get the best available LLM model based on configured API keys."""
        if self.deepseek_api_key:
            return "deepseek/deepseek-chat"
        elif self.openai_api_key:
            return "gpt-4-turbo-preview"
        elif self.anthropic_api_key:
            return "claude-3-sonnet-20240229"
        else:
            return f"ollama/{self.ollama_model}"


# Global settings instance
settings = Settings()
