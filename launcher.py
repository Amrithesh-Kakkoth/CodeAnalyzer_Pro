#!/usr/bin/env python3
"""
One-click launcher for Code Quality Intelligence Agent.
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    try:
        import code_quality_agent
        print("✅ Code Quality Agent package found")
    except ImportError:
        print("❌ Code Quality Agent package not found")
        return False
    
    # Check for required packages
    required_packages = [
        'langchain', 'langchain_community', 'langchain_groq',
        'groq', 'chromadb', 'sentence_transformers',
        'fastapi', 'uvicorn', 'streamlit', 'rich', 'click'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ All packages installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages")
            return False
    
    return True

def check_api_key():
    """Check if API key is configured."""
    print("\n🔑 Checking API configuration...")
    
    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️ No .env file found. Creating one...")
        create_env_file()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if groq_key and groq_key != "your_groq_api_key_here":
        print("✅ Groq API key configured")
        return True
    elif openai_key and openai_key != "your_openai_api_key_here":
        print("✅ OpenAI API key configured")
        return True
    else:
        print("❌ No API key configured")
        print("\n🔧 Setting up API key...")
        setup_api_key()
        return True

def create_env_file():
    """Create a basic .env file."""
    env_content = """# LLM Provider Configuration
LLM_PROVIDER=groq

# Groq API Configuration (Free tier available)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=llama-3.1-8b-instant

# OpenAI API Configuration (Paid - optional fallback)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-3.5-turbo

# Analysis Configuration
MAX_FILE_SIZE_MB=10
SUPPORTED_LANGUAGES=python,javascript,typescript
DEFAULT_SEVERITY_THRESHOLD=medium

# RAG Configuration
VECTOR_DB_PATH=./data/vector_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Web Deployment (optional)
WEB_HOST=localhost
WEB_PORT=8000
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    print("✅ Created .env file")

def setup_api_key():
    """Guide user through API key setup."""
    print("\n🌐 API Key Setup")
    print("=" * 40)
    print("1. Go to https://console.groq.com/")
    print("2. Sign up for a free account")
    print("3. Get your API key from the dashboard")
    print("4. Paste it below")
    
    api_key = input("\nEnter your Groq API key: ").strip()
    
    if api_key:
        # Update .env file
        env_file = Path(".env")
        if env_file.exists():
            content = env_file.read_text()
            content = content.replace("your_groq_api_key_here", api_key)
            env_file.write_text(content)
            print("✅ API key saved to .env file")
        else:
            print("❌ .env file not found")
    else:
        print("⚠️ No API key provided. You can set it later in the .env file")

def start_web_server():
    """Start the web server."""
    print("\n🚀 Starting Code Quality Intelligence Agent...")
    print("=" * 50)
    
    try:
        # Start web server
        subprocess.run([
            sys.executable, "-m", "code_quality_agent", "web", "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    """Main launcher function."""
    print("🎯 Code Quality Intelligence Agent Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return
    
    # Check API key
    if not check_api_key():
        print("❌ API key setup failed")
        return
    
    print("\n✅ All checks passed!")
    print("\n🌐 Starting web interface...")
    print("The web interface will open in your browser at http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    
    # Wait a moment then open browser
    time.sleep(2)
    try:
        webbrowser.open("http://localhost:8000")
    except:
        pass
    
    # Start web server
    start_web_server()

if __name__ == "__main__":
    main()


