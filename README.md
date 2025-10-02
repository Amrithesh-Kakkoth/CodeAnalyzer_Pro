# CodeAnalyzer Pro

An AI-powered code analysis tool that provides comprehensive code quality insights, automated issue detection, and intelligent Q&A capabilities using an advanced code-aware RAG system with a professional CLI interface.

## ✨ Features

- **Multi-Language Support**: Python, JavaScript, TypeScript
- **AI-Powered Analysis**: Advanced code quality assessment with AST parsing
- **Professional CLI Interface**: Beautiful, interactive command-line experience
- **Interactive Q&A**: Chat with AI about your codebase
- **GitHub Integration**: Analyze remote repositories directly from URLs
- **Enhanced RAG System**: Advanced code-aware RAG with AST parsing, entity extraction, and relationship mapping
- **Docker Support**: Containerized deployment with memory optimization
- **Free LLM Support**: Powered by Groq (free tier) with OpenAI fallback

## Images
![screenshot](image/image4.png)
![screenshot2](image/image2.png)
![screenshot3](image/image3.png)

## 🚀 Quick Start

### Quick Setup & Run

#### Option 1: Docker Compose (Easiest!)
```bash
# One command to run everything
docker-compose up code-analyzer-pro
```

#### Option 2: Simple Docker Run
```bash
# Run with pre-built image
docker run --rm -it code-analyzer-pro \
  python -m code_quality_agent chat https://github.com/username/repo
```

#### Option 3: Local Installation
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key** (or use setup script):
   ```bash
   python setup.py  # Interactive setup
   ```

3. **Start interactive chat**:
   ```bash
   python -m code_quality_agent chat https://github.com/username/repo
   ```
   
   **Quick Run**: `python -m code_quality_agent chat <your repo name>`

## ⚙️ Setup

### Quick Configuration
```bash
# Interactive setup script (recommended)
python setup.py
```

The setup script will:
- 🤖 Ask for LLM provider (Groq/OpenAI)
- 🔑 Request API keys
- 🧠 Let you choose models
- ⚙️ Configure analysis settings
- 💾 Create `.env` file automatically

### Manual Configuration
Create `.env` file manually:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_key_here
LLM_PROVIDER=groq
```

## 📖 Usage

### Interactive CLI (Primary Interface)

#### Professional Chat Interface
```bash
# Analyze GitHub repository with interactive chat
python -m code_quality_agent chat https://github.com/username/repo

# Analyze local codebase with interactive chat
python -m code_quality_agent chat /path/to/code
```

**Features:**
- 🎨 **Beautiful CLI UI** with professional panels and colors
- 💬 **Interactive Q&A** - Ask natural language questions
- 🔍 **Built-in Commands** - `analyze`, `security`, `performance`, `help`
- 📊 **Real-time Analysis** - Live code quality metrics
- 🚀 **GitHub Integration** - Direct URL analysis

#### Available Commands in Chat
- `analyze` - Run comprehensive code quality analysis
- `security` - Check for security vulnerabilities
- `performance` - Identify performance bottlenecks
- `complexity` - Analyze code complexity
- `documentation` - Review documentation gaps
- `testing` - Assess testing coverage
- `help` - Show detailed command reference
- `quit` - Exit the application

### Command Line Analysis

#### Analyze Local Code
```bash
# Analyze a directory
python -m code_quality_agent analyze ./my-project

# Analyze with enhanced mode
python -m code_quality_agent analyze ./my-project --enhanced

# Save results to file
python -m code_quality_agent analyze ./my-project --output results.json
```

#### Analyze GitHub Repositories
```bash
# Analyze a GitHub repository
python -m code_quality_agent analyze https://github.com/user/repo

# Analyze specific branch
python -m code_quality_agent analyze https://github.com/user/repo@develop

# Search for popular repositories
python -m code_quality_agent search python --limit 10
```

#### Interactive Q&A
```bash
# Chat with AI about your code
python -m code_quality_agent chat ./my-project
```

## 🔧 Configuration

Create a `.env` file in the project root:

```env
# LLM Provider Configuration
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

## 🏗️ Architecture

### Core Components
- **`analyzer.py`**: Main code analysis engine
- **`qa_agent.py`**: AI-powered Q&A system
- **`rag_system.py`**: Retrieval-Augmented Generation
- **`github_analyzer.py`**: GitHub repository integration
- **`interactive_cli.py`**: Professional CLI interface
- **`cli.py`**: Command-line interface

### AI Integration
- **Groq**: Free LLM for Q&A and analysis
- **OpenAI**: Optional paid fallback
- **Local Embeddings**: Free vector embeddings
- **LangChain**: AI framework integration

## 🏗️ Architecture & Improvements

### Enhanced CLI Interface
- **Professional UI Design**: Beautiful panels with Rich library styling
- **Interactive Commands**: Built-in commands for analysis, security, performance
- **Real-time Feedback**: Live progress indicators and status updates
- **Color-coded Output**: Professional color scheme for better readability

### Advanced Code-Aware RAG System
- **AST-Based Understanding**: Deep code structure analysis beyond simple text processing
- **Entity Extraction**: Intelligent identification of functions, classes, variables, and imports
- **Relationship Mapping**: Code dependency tracking, call chains, and inheritance analysis
- **Enhanced Metadata**: Rich contextual information including line numbers, dependencies, and callers
- **Semantic Search**: Vector similarity search enhanced with code relationship context
- **Source Attribution**: Precise file, line, and function-level source tracking

### Memory Optimization
- **Docker Memory Limits**: Optimized for 1GB RAM usage
- **CPU-only PyTorch**: Reduced memory footprint
- **Environment Variables**: Memory optimization settings
- **Lightweight Dependencies**: Minimal package requirements

### GitHub Integration
- **Direct URL Analysis**: Analyze repositories without cloning locally
- **Automatic Download**: Temporary repository cloning and cleanup
- **Repository Metadata**: Stars, forks, language detection
- **Path Resolution**: Automatic path handling for downloaded repos

## 📊 Analysis Features

### Code Quality Metrics
- **Security Issues**: Vulnerabilities and security risks
- **Performance Problems**: Bottlenecks and optimization opportunities
- **Code Smells**: Maintainability issues
- **Testing Gaps**: Missing test coverage
- **Documentation Issues**: Incomplete or missing docs
- **Complexity Analysis**: Cyclomatic complexity and code complexity

### Issue Detection
- **Automated Scanning**: AST-based analysis
- **Pattern Recognition**: Common anti-patterns
- **Best Practices**: Coding standard violations
- **Severity Scoring**: P0-P4 priority levels

## 🐳 Docker Deployment

### Containerized Setup
- **Memory Optimized**: Runs efficiently with 1GB RAM
- **Environment Isolation**: Clean, reproducible environment
- **Volume Mounting**: Access to local code and configuration
- **Multi-platform**: Works on Windows, macOS, and Linux

### Docker Commands

#### Simple Usage (Recommended)
```bash
# Use Docker Compose (easiest)
docker-compose up code-analyzer-pro

# Or simple Docker run
docker run --rm -it code-analyzer-pro \
  python -m code_quality_agent chat https://github.com/username/repo
```

#### Advanced Usage (if needed)
```bash
# Build the image
docker build -t code-analyzer-pro .

# Run with custom settings
docker run --rm -it --memory=1g --memory-swap=1g \
  -v ${PWD}:/workspace:ro \
  -v ${PWD}/.env:/app/.env:ro \
  code-analyzer-pro \
  python -m code_quality_agent chat https://github.com/username/repo
```

## 🔍 Enhanced Code-Aware RAG System

Unlike vanilla RAG systems that treat code as simple text, our system provides deep code understanding:

### **Beyond Vanilla RAG**
- **Code Structure Analysis**: AST parsing understands syntax and semantics
- **Entity-Aware Processing**: Identifies functions, classes, variables, and their relationships
- **Dependency Tracking**: Maps how code pieces connect and depend on each other
- **Contextual Retrieval**: Finds relevant code based on meaning, not just keywords

### **Key Capabilities**
- **Code Indexing**: Vector embeddings with rich metadata and structural information
- **Semantic Search**: Find relevant code sections based on functionality and relationships
- **Context-Aware Q&A**: AI understands your codebase structure and can explain how components work together
- **Source Attribution**: Precise tracking of which files, lines, and functions provide information
- **Local Processing**: No external API calls for embeddings - all processing happens locally

## 📝 Examples

### Analyze a Python Project
```bash
python -m code_quality_agent analyze ./my-python-project --enhanced
```

### Chat About Your Code
```bash
python -m code_quality_agent chat ./my-project

# Ask sophisticated questions that showcase enhanced RAG:
# "What are the main functions in this code?"
# "How does the authentication system work?"
# "What are the dependencies between modules?"
# "Show me the call chain for user registration"
# "What functions depend on the database connection?"
# "How can I improve the performance of this specific function?"
# "What security vulnerabilities exist in the auth module?"
# "Explain the data flow from API request to database storage"
```

### Analyze GitHub Repository
```bash
python -m code_quality_agent analyze https://github.com/microsoft/vscode-python
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Groq**: For providing free LLM access
- **LangChain**: For AI framework capabilities
- **ChromaDB**: For vector storage
- **Rich**: For beautiful CLI output

## 📞 Support

For issues and questions:
1. Check the documentation
2. Search existing issues
3. Create a new issue with details
4. Join our community discussions

---

**Made with ❤️ for developers who care about code quality**
