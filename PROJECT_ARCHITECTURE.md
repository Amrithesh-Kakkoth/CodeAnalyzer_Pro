# 🏗️ CodeAnalyzer Pro - Project Architecture

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [RAG Architecture](#rag-architecture)
4. [AST Parsing Architecture](#ast-parsing-architecture)
5. [Data Pipeline](#data-pipeline)
6. [Component Details](#component-details)
7. [Technology Stack](#technology-stack)

---

## 🎯 Project Overview

The **CodeAnalyzer Pro** is an AI-powered system that analyzes code repositories to provide comprehensive quality insights, security assessments, and architectural recommendations. It goes beyond traditional linting by understanding code relationships, dependencies, and providing conversational Q&A capabilities.

### Key Features
- **Multi-language Support**: Python, JavaScript, TypeScript
- **AST Parsing**: Deep structural analysis of code
- **RAG System**: Retrieval-Augmented Generation for context-aware responses
- **Professional CLI Interface**: Beautiful, interactive command-line experience
- **Security Analysis**: Vulnerability detection and risk assessment
- **Performance Analysis**: Bottleneck identification and optimization suggestions
- **Interactive Q&A**: Natural language queries about codebase
- **GitHub Integration**: Direct repository analysis from URLs
- **Docker Support**: Containerized deployment with memory optimization
- **Memory Optimization**: Efficient resource usage with 1GB RAM limits

---

## 🏛️ System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        CLI[CLI Interface]
        INTERACTIVE[Interactive Chat]
    end
    
    subgraph "Core Application Layer"
        QA_AGENT[Q&A Agent]
        ANALYZER[Code Analyzer]
        CONFIG[Configuration Manager]
    end
    
    subgraph "Analysis Engine"
        ENHANCED_RAG[Enhanced RAG System]
        AST_PARSER[AST Parser]
        PATTERN_MATCHER[Pattern Matcher]
        SEVERITY_SCORER[Severity Scorer]
    end
    
    subgraph "Data Processing"
        GITHUB_ANALYZER[GitHub Analyzer]
        LANGUAGE_DETECTOR[Language Detector]
        CODE_STRUCTURE[Code Structure Analyzer]
    end
    
    subgraph "Storage Layer"
        VECTOR_DB[(Vector Database)]
        CHROMA_DB[(ChromaDB)]
        FILE_SYSTEM[(File System)]
    end
    
    subgraph "External Services"
        OPENAI[OpenAI API]
        GROQ[Groq API]
        GITHUB_API[GitHub API]
    end
    
    CLI --> QA_AGENT
    INTERACTIVE --> QA_AGENT
    QA_AGENT --> ENHANCED_RAG
    QA_AGENT --> ANALYZER
    ANALYZER --> AST_PARSER
    ANALYZER --> PATTERN_MATCHER
    ANALYZER --> SEVERITY_SCORER
    ENHANCED_RAG --> CODE_STRUCTURE
    ENHANCED_RAG --> VECTOR_DB
    GITHUB_ANALYZER --> GITHUB_API
    QA_AGENT --> OPENAI
    QA_AGENT --> GROQ
    VECTOR_DB --> CHROMA_DB
    CODE_STRUCTURE --> FILE_SYSTEM
```

---

## 🖥️ CLI Architecture

### Enhanced Interactive CLI Design

The new CLI interface provides a professional, user-friendly experience with the following architectural improvements:

```mermaid
graph TB
    subgraph "CLI Interface Layer"
        WELCOME[Welcome Screen]
        HELP[Help System]
        INPUT[Input Handler]
        OUTPUT[Output Renderer]
    end
    
    subgraph "Command Processing"
        COMMAND_DETECTOR[Command Detector]
        ANALYZE_CMD[Analyze Command]
        CHAT_CMD[Chat Commands]
        UTILITY_CMD[Utility Commands]
    end
    
    subgraph "UI Components"
        PANELS[Rich Panels]
        TABLES[Data Tables]
        PROGRESS[Progress Indicators]
        COLORS[Color Schemes]
    end
    
    subgraph "Path Resolution"
        GITHUB_PATH[GitHub Path Handler]
        LOCAL_PATH[Local Path Handler]
        TEMP_CLEANUP[Temp Cleanup]
    end
    
    WELCOME --> PANELS
    HELP --> PANELS
    INPUT --> COMMAND_DETECTOR
    COMMAND_DETECTOR --> ANALYZE_CMD
    COMMAND_DETECTOR --> CHAT_CMD
    COMMAND_DETECTOR --> UTILITY_CMD
    ANALYZE_CMD --> GITHUB_PATH
    ANALYZE_CMD --> LOCAL_PATH
    OUTPUT --> TABLES
    OUTPUT --> PROGRESS
    OUTPUT --> COLORS
    GITHUB_PATH --> TEMP_CLEANUP
```

### Key Architectural Improvements

#### 1. Professional UI Design
- **Rich Library Integration**: Beautiful panels, tables, and progress indicators
- **Color-coded Output**: Professional color scheme for better readability
- **Responsive Layout**: Adaptive panels based on terminal size
- **Consistent Styling**: Unified design language throughout

#### 2. Command Processing Architecture
- **Command Detection**: Smart detection of commands vs. natural language questions
- **Path Resolution**: Automatic handling of GitHub URLs vs. local paths
- **Error Handling**: Graceful error display with professional formatting
- **State Management**: Proper initialization and cleanup of components

#### 3. Memory Optimization
- **Docker Memory Limits**: Optimized for 1GB RAM usage
- **Environment Variables**: Memory optimization settings
- **Resource Cleanup**: Automatic cleanup of temporary files
- **Efficient Processing**: Streamlined operations to reduce memory footprint

#### 4. GitHub Integration
- **Direct URL Analysis**: Seamless GitHub repository analysis
- **Automatic Download**: Temporary repository cloning and cleanup
- **Path Resolution**: Automatic path handling for downloaded repositories
- **Metadata Extraction**: Repository information display

---

## 🔍 RAG Architecture

```mermaid
graph TB
    subgraph "Input Processing"
        CODE_INPUT[Code Repository]
        GITHUB_URL[GitHub URL]
        LOCAL_PATH[Local Path]
    end
    
    subgraph "Code Analysis"
        AST_ANALYZER[AST Analyzer]
        ENTITY_EXTRACTOR[Entity Extractor]
        RELATIONSHIP_MAPPER[Relationship Mapper]
    end
    
    subgraph "Document Processing"
        DOC_CREATOR[Document Creator]
        METADATA_ENHANCER[Metadata Enhancer]
        TEXT_SPLITTER[Text Splitter]
    end
    
    subgraph "Vector Store"
        EMBEDDINGS[Embeddings Generator]
        CHROMA_STORE[Chroma Vector Store]
        SIMILARITY_SEARCH[Similarity Search]
    end
    
    subgraph "Retrieval & Generation"
        QUERY_PROCESSOR[Query Processor]
        CONTEXT_RETRIEVER[Context Retriever]
        LLM_GENERATOR[LLM Generator]
    end
    
    subgraph "Output"
        ENHANCED_RESPONSE[Enhanced Response]
        SOURCE_ATTRIBUTION[Source Attribution]
    end
    
    CODE_INPUT --> AST_ANALYZER
    GITHUB_URL --> AST_ANALYZER
    LOCAL_PATH --> AST_ANALYZER
    
    AST_ANALYZER --> ENTITY_EXTRACTOR
    ENTITY_EXTRACTOR --> RELATIONSHIP_MAPPER
    
    RELATIONSHIP_MAPPER --> DOC_CREATOR
    DOC_CREATOR --> METADATA_ENHANCER
    METADATA_ENHANCER --> TEXT_SPLITTER
    
    TEXT_SPLITTER --> EMBEDDINGS
    EMBEDDINGS --> CHROMA_STORE
    
    QUERY_PROCESSOR --> SIMILARITY_SEARCH
    SIMILARITY_SEARCH --> CHROMA_STORE
    SIMILARITY_SEARCH --> CONTEXT_RETRIEVER
    
    CONTEXT_RETRIEVER --> LLM_GENERATOR
    LLM_GENERATOR --> ENHANCED_RESPONSE
    ENHANCED_RESPONSE --> SOURCE_ATTRIBUTION
```

### RAG Components

1. **Code Analysis Pipeline**
   - AST parsing for structural understanding
   - Entity extraction (functions, classes, imports)
   - Relationship mapping (dependencies, calls)

2. **Document Processing**
   - Enhanced metadata with filename, language, AST info
   - Code content inclusion
   - Text chunking for optimal retrieval

3. **Vector Store**
   - ChromaDB for similarity search
   - Sentence Transformers for embeddings
   - Complex metadata filtering

4. **Retrieval & Generation**
   - Context-aware query processing
   - Multi-source retrieval
   - LLM integration with source attribution

---

## 🌳 AST Parsing Architecture

```mermaid
graph TB
    subgraph "Input Files"
        PYTHON_FILES[Python Files .py]
        JS_FILES[JavaScript Files .js]
        TS_FILES[TypeScript Files .ts]
    end
    
    subgraph "Language Detection"
        EXTENSION_CHECKER[Extension Checker]
        LANGUAGE_MAPPER[Language Mapper]
    end
    
    subgraph "Python AST Processing"
        PYTHON_AST[Python AST Module]
        FUNCTION_VISITOR[Function Visitor]
        CLASS_VISITOR[Class Visitor]
        IMPORT_VISITOR[Import Visitor]
    end
    
    subgraph "JS/TS Processing"
        REGEX_PARSER[Regex Parser]
        FUNCTION_REGEX[Function Regex]
        CLASS_REGEX[Class Regex]
        IMPORT_REGEX[Import Regex]
    end
    
    subgraph "Entity Extraction"
        CODE_ENTITY[Code Entity]
        ENTITY_METADATA[Entity Metadata]
        LINE_NUMBERS[Line Numbers]
    end
    
    subgraph "Relationship Analysis"
        DEPENDENCY_TRACKER[Dependency Tracker]
        CALL_GRAPH[Call Graph]
        RELATIONSHIP_MAPPER[Relationship Mapper]
    end
    
    subgraph "Output"
        ENTITIES_LIST[Entities List]
        RELATIONSHIPS_LIST[Relationships List]
        METADATA_STORE[Metadata Store]
    end
    
    PYTHON_FILES --> EXTENSION_CHECKER
    JS_FILES --> EXTENSION_CHECKER
    TS_FILES --> EXTENSION_CHECKER
    
    EXTENSION_CHECKER --> LANGUAGE_MAPPER
    
    LANGUAGE_MAPPER --> PYTHON_AST
    LANGUAGE_MAPPER --> REGEX_PARSER
    
    PYTHON_AST --> FUNCTION_VISITOR
    PYTHON_AST --> CLASS_VISITOR
    PYTHON_AST --> IMPORT_VISITOR
    
    REGEX_PARSER --> FUNCTION_REGEX
    REGEX_PARSER --> CLASS_REGEX
    REGEX_PARSER --> IMPORT_REGEX
    
    FUNCTION_VISITOR --> CODE_ENTITY
    CLASS_VISITOR --> CODE_ENTITY
    IMPORT_VISITOR --> CODE_ENTITY
    FUNCTION_REGEX --> CODE_ENTITY
    CLASS_REGEX --> CODE_ENTITY
    IMPORT_REGEX --> CODE_ENTITY
    
    CODE_ENTITY --> ENTITY_METADATA
    ENTITY_METADATA --> LINE_NUMBERS
    
    LINE_NUMBERS --> DEPENDENCY_TRACKER
    DEPENDENCY_TRACKER --> CALL_GRAPH
    CALL_GRAPH --> RELATIONSHIP_MAPPER
    
    RELATIONSHIP_MAPPER --> ENTITIES_LIST
    RELATIONSHIP_MAPPER --> RELATIONSHIPS_LIST
    RELATIONSHIP_MAPPER --> METADATA_STORE
```

### AST Processing Details

1. **Language Detection**
   - File extension mapping
   - Language-specific processing paths

2. **Python AST Processing**
   - Native `ast` module usage
   - Visitor pattern for traversal
   - Function, class, and import extraction

3. **JavaScript/TypeScript Processing**
   - Regex-based parsing
   - Pattern matching for functions, classes
   - Import statement detection

4. **Entity Extraction**
   - Name, type, line number extraction
   - Docstring and parameter parsing
   - Dependency identification

5. **Relationship Analysis**
   - Call graph construction
   - Dependency mapping
   - Cross-reference tracking

---

## 🔄 Data Pipeline

```mermaid
flowchart TD
    subgraph "Data Ingestion"
        A[Code Repository Input]
        B[GitHub URL Input]
        C[Local File System]
    end
    
    subgraph "Preprocessing"
        D[File Discovery]
        E[Language Detection]
        F[Content Extraction]
    end
    
    subgraph "Analysis Phase"
        G[AST Parsing]
        H[Pattern Matching]
        I[Entity Extraction]
        J[Relationship Mapping]
    end
    
    subgraph "Quality Assessment"
        K[Security Analysis]
        L[Performance Analysis]
        M[Complexity Analysis]
        N[Documentation Analysis]
    end
    
    subgraph "Data Storage"
        O[Vector Database]
        P[Metadata Store]
        Q[Analysis Results]
    end
    
    subgraph "RAG Processing"
        R[Document Creation]
        S[Embedding Generation]
        T[Vector Indexing]
    end
    
    subgraph "Query Processing"
        U[User Query]
        V[Context Retrieval]
        W[Response Generation]
    end
    
    A --> D
    B --> D
    C --> D
    
    D --> E
    E --> F
    
    F --> G
    F --> H
    G --> I
    H --> I
    I --> J
    
    I --> K
    I --> L
    I --> M
    I --> N
    
    J --> R
    R --> S
    S --> T
    T --> O
    
    K --> Q
    L --> Q
    M --> Q
    N --> Q
    
    O --> P
    Q --> P
    
    U --> V
    V --> O
    V --> W
    W --> U
```

### Pipeline Stages

1. **Data Ingestion**
   - Multiple input sources (local, GitHub)
   - File discovery and filtering
   - Content extraction

2. **Preprocessing**
   - Language detection
   - File type validation
   - Content normalization

3. **Analysis Phase**
   - AST parsing for structure
   - Pattern matching for issues
   - Entity and relationship extraction

4. **Quality Assessment**
   - Security vulnerability detection
   - Performance bottleneck identification
   - Complexity analysis
   - Documentation gap analysis

5. **Data Storage**
   - Vector database for RAG
   - Metadata storage
   - Analysis result persistence

6. **RAG Processing**
   - Document creation with metadata
   - Embedding generation
   - Vector indexing

7. **Query Processing**
   - Natural language query processing
   - Context retrieval
   - AI-powered response generation

---

## 🧩 Component Details

### Core Components

1. **Q&A Agent** (`qa_agent.py`)
   - LangChain integration
   - Conversation memory
   - Multi-LLM support (OpenAI, Groq)

2. **Enhanced RAG System** (`enhanced_rag_system.py`)
   - ChromaDB vector store
   - Sentence Transformers embeddings
   - Complex metadata handling

3. **Code Analyzer** (`analyzer.py`)
   - Multi-language analysis
   - Issue detection and categorization
   - Severity scoring

4. **AST Parser** (`enhanced_rag_system.py`)
   - Python AST module
   - JavaScript/TypeScript regex parsing
   - Entity relationship mapping

5. **GitHub Analyzer** (`github_analyzer.py`)
   - Repository cloning
   - API integration
   - File structure analysis

6. **Severity Scorer** (`severity_scorer.py`)
   - AI-powered impact assessment
   - Likelihood evaluation
   - Priority ranking

### Data Models

1. **CodeEntity**
   - Name, type, line number
   - Docstring, parameters
   - Dependencies

2. **CodeRelationship**
   - Source and target entities
   - Relationship type
   - Line number reference

3. **CodeIssue**
   - Issue type and description
   - Severity and priority
   - Suggested fixes

---

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.11+**: Main programming language
- **LangChain**: LLM framework and orchestration
- **ChromaDB**: Vector database for RAG
- **Sentence Transformers**: Embedding generation

### AI/ML Libraries
- **OpenAI API**: GPT models for analysis
- **Groq API**: Fast inference for Q&A
- **HuggingFace**: Embedding models

### Analysis Tools
- **AST Module**: Python code parsing
- **Regex**: JavaScript/TypeScript parsing
- **NetworkX**: Graph analysis
- **Matplotlib/Seaborn**: Visualization

### Infrastructure
- **Docker**: Containerization
- **Click**: CLI framework
- **Rich**: Terminal UI
- **Pydantic**: Configuration management

### Development Tools
- **Git**: Version control
- **Docker Compose**: Orchestration
- **Pytest**: Testing framework

---

## 🚀 Deployment Architecture

```mermaid
graph TB
    subgraph "Development Environment"
        DEV[Developer Machine]
        LOCAL_DOCKER[Local Docker]
        TEST_DATA[Test Data]
    end
    
    subgraph "Container Layer"
        DOCKER_IMAGE[Docker Image]
        CONTAINER[Container Instance]
        VOLUMES[Volume Mounts]
    end
    
    subgraph "Application Layer"
        CLI_INTERFACE[CLI Interface]
        RAG_SYSTEM[RAG System]
        ANALYSIS_ENGINE[Analysis Engine]
    end
    
    subgraph "Data Layer"
        VECTOR_DB[(Vector Database)]
        CONFIG_FILES[Configuration Files]
        CODE_REPOS[Code Repositories]
    end
    
    subgraph "External Services"
        API_SERVICES[AI APIs]
        GITHUB_API[GitHub API]
    end
    
    DEV --> LOCAL_DOCKER
    LOCAL_DOCKER --> DOCKER_IMAGE
    DOCKER_IMAGE --> CONTAINER
    CONTAINER --> VOLUMES
    
    CONTAINER --> CLI_INTERFACE
    CLI_INTERFACE --> RAG_SYSTEM
    CLI_INTERFACE --> ANALYSIS_ENGINE
    
    RAG_SYSTEM --> VECTOR_DB
    ANALYSIS_ENGINE --> CONFIG_FILES
    VOLUMES --> CODE_REPOS
    
    RAG_SYSTEM --> API_SERVICES
    ANALYSIS_ENGINE --> GITHUB_API
```

This architecture ensures scalability, maintainability, and ease of deployment while providing comprehensive code analysis capabilities.
