# Code Quality Intelligence Agent

An AI-powered code analysis tool that provides comprehensive code quality insights, automated issue detection, and interactive Q&A capabilities.

## ✨ Features

- **Multi-Language Support**: Python, JavaScript, TypeScript
- **AI-Powered Analysis**: Advanced code quality assessment
- **Interactive Q&A**: Chat with AI about your codebase
- **GitHub Integration**: Analyze remote repositories
- **Web Interface**: User-friendly web dashboard
- **RAG System**: Intelligent code understanding
- **Free LLM Support**: Powered by Groq (free tier)

## 🚀 Quick Start

### One-Click Launch
```bash
python launcher.py
```

### Manual Setup
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key**:
   - Get a free API key from [Groq Console](https://console.groq.com/)
   - Set it in your `.env` file:
     ```env
     GROQ_API_KEY=your_groq_api_key_here
     ```

3. **Start the web interface**:
   ```bash
   python -m code_quality_agent web
   ```

## 📖 Usage

### Web Interface
- Open `http://localhost:8000` in your browser
- Upload code files or analyze GitHub repositories
- Chat with AI about your code

### Command Line

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

# Web Deployment
WEB_HOST=localhost
WEB_PORT=8000
```

## 🏗️ Architecture

### Core Components
- **`analyzer.py`**: Main code analysis engine
- **`qa_agent.py`**: AI-powered Q&A system
- **`rag_system.py`**: Retrieval-Augmented Generation
- **`github_analyzer.py`**: GitHub repository integration
- **`web_app.py`**: FastAPI web interface
- **`cli.py`**: Command-line interface

### AI Integration
- **Groq**: Free LLM for Q&A and analysis
- **OpenAI**: Optional paid fallback
- **Local Embeddings**: Free vector embeddings
- **LangChain**: AI framework integration

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

## 🌐 Web Interface

The web interface provides:
- **File Upload**: Drag-and-drop code analysis
- **Real-time Results**: Instant quality reports
- **Interactive Chat**: AI-powered code Q&A
- **Visualization**: Charts and graphs
- **Export Options**: JSON, HTML, Markdown

## 🔍 RAG System

The Retrieval-Augmented Generation system:
- **Code Indexing**: Vector embeddings of code
- **Semantic Search**: Find relevant code sections
- **Context-Aware Q&A**: AI understands your codebase
- **Local Processing**: No external API calls for embeddings

## 📝 Examples

### Analyze a Python Project
```bash
python -m code_quality_agent analyze ./my-python-project --enhanced
```

### Chat About Your Code
```bash
python -m code_quality_agent chat ./my-project
# Ask: "What are the main functions in this code?"
# Ask: "How can I improve the performance?"
# Ask: "What security issues do you see?"
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
- **FastAPI**: For web framework
- **Rich**: For beautiful CLI output

## 📞 Support

For issues and questions:
1. Check the documentation
2. Search existing issues
3. Create a new issue with details
4. Join our community discussions

---

**Made with ❤️ for developers who care about code quality**