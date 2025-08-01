import os
from typing import Optional
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file if it exists"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # Fallback to manual loading if python-dotenv not available
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value

# Load .env file
load_env_file()

# Simple settings class without Pydantic dependency
class Settings:
    """Application settings with environment variable support"""
    
    def __init__(self):
        # API Configuration
        self.claude_api_key = os.getenv("CLAUDE_API_KEY", "")
        self.claude_model = "claude-3-5-sonnet-20241022"
        
        # Processing Configuration
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "4000"))
        self.overlap_size = int(os.getenv("OVERLAP_SIZE", "200"))
        self.default_confidence_threshold = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.5"))
        
        # API Configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.api_reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    def validate_api_key(self):
        """Validate that API key is set and not placeholder"""
        if not self.claude_api_key or self.claude_api_key == "your-actual-claude-api-key-here":
            raise ValueError(
                "Claude API key not found! Please:\n"
                "1. Create a .env file in the project directory\n"
                "2. Add: CLAUDE_API_KEY=your-actual-api-key\n"
                "3. Replace 'your-actual-api-key' with your real Claude API key"
            )
        return self.claude_api_key

settings = Settings()