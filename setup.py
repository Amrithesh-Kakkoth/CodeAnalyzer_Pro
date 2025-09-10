#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for CodeAnalyzer Pro
Creates .env file with user-provided configuration
"""

import os
import sys
from pathlib import Path
from typing import Optional


def get_input(prompt: str, default: Optional[str] = None, required: bool = True) -> str:
    """Get user input with optional default value."""
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "
    
    while True:
        value = input(full_prompt).strip()
        if value:
            return value
        elif default:
            return default
        elif not required:
            return ""
        else:
            print("❌ This field is required. Please enter a value.")


def get_choice(prompt: str, choices: list, default: Optional[str] = None) -> str:
    """Get user choice from a list of options."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        marker = " (default)" if choice == default else ""
        print(f"  {i}. {choice}{marker}")
    
    while True:
        try:
            choice_input = input(f"\nEnter your choice (1-{len(choices)}): ").strip()
            if not choice_input and default:
                return default
            
            choice_num = int(choice_input)
            if 1 <= choice_num <= len(choices):
                return choices[choice_num - 1]
            else:
                print(f"❌ Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print("❌ Please enter a valid number")


def create_env_file(config: dict) -> None:
    """Create .env file with the provided configuration."""
    env_content = f"""# LLM Provider Configuration
LLM_PROVIDER={config['llm_provider']}

# Groq API Configuration (Free tier available)
GROQ_API_KEY={config['groq_api_key']}
GROQ_MODEL_NAME={config['groq_model']}

# OpenAI API Configuration (Paid - optional fallback)
OPENAI_API_KEY={config['openai_api_key']}

# Analysis Configuration
MAX_FILE_SIZE_MB={config['max_file_size']}
SUPPORTED_LANGUAGES={config['supported_languages']}
DEFAULT_SEVERITY_THRESHOLD={config['severity_threshold']}

# RAG Configuration
VECTOR_DB_PATH=./data/vector_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Web Deployment (optional)
WEB_HOST=localhost
WEB_PORT=8000

# Memory Optimization
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
"""
    
    env_file = Path(".env")
    env_file.write_text(env_content)
    print(f"✅ Created .env file at {env_file.absolute()}")


def main():
    """Main setup function."""
    print("🚀 CodeAnalyzer Pro Setup")
    print("=" * 50)
    
    # Check if .env already exists
    if Path(".env").exists():
        overwrite = get_input(
            "⚠️  .env file already exists. Overwrite?", 
            default="n", 
            required=False
        ).lower()
        if overwrite not in ['y', 'yes']:
            print("❌ Setup cancelled.")
            return
    
    print("\n📋 Configuration Setup")
    print("-" * 30)
    
    # LLM Provider selection
    llm_providers = ["groq", "openai"]
    llm_provider = get_choice(
        "🤖 Select LLM Provider:",
        llm_providers,
        default="groq"
    )
    
    # Groq configuration
    print(f"\n🔑 {llm_provider.upper()} Configuration")
    print("-" * 30)
    
    if llm_provider == "groq":
        groq_api_key = get_input(
            "🔑 Enter your Groq API Key",
            required=True
        )
        
        groq_models = [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile", 
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        groq_model = get_choice(
            "🧠 Select Groq Model:",
            groq_models,
            default="llama-3.1-8b-instant"
        )
        
        openai_api_key = get_input(
            "🔑 Enter OpenAI API Key (optional fallback)",
            required=False
        )
    else:
        openai_api_key = get_input(
            "🔑 Enter your OpenAI API Key",
            required=True
        )
        groq_api_key = get_input(
            "🔑 Enter Groq API Key (optional fallback)",
            required=False
        )
        groq_model = "llama-3.1-8b-instant"
    
    # Analysis configuration
    print(f"\n⚙️  Analysis Configuration")
    print("-" * 30)
    
    max_file_size = get_input(
        "📁 Maximum file size for analysis (MB)",
        default="10",
        required=False
    )
    
    supported_languages = get_input(
        "🌐 Supported languages (comma-separated)",
        default="python,javascript,typescript",
        required=False
    )
    
    severity_levels = ["low", "medium", "high", "critical"]
    severity_threshold = get_choice(
        "⚠️  Default severity threshold:",
        severity_levels,
        default="medium"
    )
    
    # Create configuration dictionary
    config = {
        'llm_provider': llm_provider,
        'groq_api_key': groq_api_key,
        'groq_model': groq_model,
        'openai_api_key': openai_api_key,
        'max_file_size': max_file_size,
        'supported_languages': supported_languages,
        'severity_threshold': severity_threshold
    }
    
    # Create .env file
    print(f"\n💾 Creating .env file...")
    create_env_file(config)
    
    # Show next steps
    print(f"\n🎉 Setup Complete!")
    print("=" * 50)
    print("📋 Next Steps:")
    print("1. Run the agent locally:")
    print("   python -m code_quality_agent chat https://github.com/username/repo")
    print("\n2. Or run with Docker:")
    print("   docker run --rm -it --memory=1g --memory-swap=1g \\")
    print("     -v ${PWD}:/workspace:ro \\")
    print("     -v ${PWD}/.env:/app/.env:ro \\")
    print("     code-quality-agent \\")
    print("     python -m code_quality_agent chat https://github.com/username/repo")
    print("\n3. Or use Docker Compose:")
    print("   docker-compose up code-quality-agent")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
