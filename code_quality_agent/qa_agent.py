"""
Interactive Q&A agent for code analysis using LangChain.
"""

import warnings
from pathlib import Path
from typing import List
from langchain_community.llms import OpenAI
from groq import Groq
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

from .config import config
from .enhanced_rag_system import EnhancedCodeRAGSystem


class CodeQAAgent:
    """AI-powered Q&A agent for code analysis."""
    
    def __init__(self, codebase_path: str, model_name: str = None, github_data: dict = None):
        """Initialize the Q&A agent with a codebase."""
        self.codebase_path = codebase_path
        self.is_github_repo = codebase_path.startswith(('http://github.com/', 'https://github.com/'))
        self.model_name = model_name or config.ai.groq_model_name
        self.github_data = github_data  # GitHub repository data for RAG
        self.llm = None
        self.vectorstore = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create unique vectorstore path for this analysis
        import uuid
        self.vectorstore_path = f"{config.rag.vector_db_path}_{uuid.uuid4().hex[:8]}"
        
        # Initialize enhanced RAG system
        if self.is_github_repo:
            # For GitHub repos, we'll download and analyze
            self._setup_github_rag()
        else:
            self.enhanced_rag = EnhancedCodeRAGSystem(str(self.codebase_path))
        
        # Initialize if API key is available
        if self._has_api_key():
            self._initialize_llm()
            if not self.is_github_repo:
                self._setup_enhanced_retrieval_chain()
    
    def _has_api_key(self):
        """Check if we have a valid API key for the selected provider."""
        if config.ai.llm_provider == "groq":
            return bool(config.ai.groq_api_key)
        elif config.ai.llm_provider == "openai":
            return bool(config.ai.openai_api_key)
        return False
    
    def _initialize_llm(self):
        """Initialize the language model based on provider."""
        if config.ai.llm_provider == "groq":
            # Use LangChain Groq wrapper for compatibility
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                groq_api_key=config.ai.groq_api_key,
                model_name=self.model_name,
                temperature=config.ai.temperature,
                max_tokens=config.ai.max_tokens
            )
            # Also keep direct client for fallback
            self.groq_client = Groq(api_key=config.ai.groq_api_key)
        elif config.ai.llm_provider == "openai":
            self.llm = OpenAI(
                openai_api_key=config.ai.openai_api_key,
                model_name=self.model_name,
                temperature=config.ai.temperature,
                max_tokens=config.ai.max_tokens
            )
    
    def _setup_enhanced_retrieval_chain(self):
        """Set up the enhanced retrieval chain for RAG."""
        try:
            if not self.enhanced_rag:
                print("No enhanced RAG system available")
                return
                
            # Index the codebase with enhanced analysis
            result = self.enhanced_rag.index_codebase()
            
            if "error" in result:
                print(f"Error indexing codebase: {result['error']}")
                # Fallback to basic RAG
                self._setup_basic_retrieval_chain()
            else:
                self.vectorstore = self.enhanced_rag.vectorstore
                print(f"Enhanced RAG initialized: {result['documents_indexed']} files, {result['entities_found']} entities, {result['relationships_found']} relationships")
            
        except Exception as e:
            print(f"Error setting up enhanced retrieval chain: {e}")
            # Fallback to basic RAG
            self._setup_basic_retrieval_chain()
    
    def _setup_basic_retrieval_chain(self):
        """Set up basic retrieval chain as fallback."""
        try:
            # Load and process documents
            documents = self._load_codebase_documents()
            
            if not documents:
                return
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.rag.chunk_size,
                chunk_overlap=config.rag.chunk_overlap
            )
            texts = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store (use local embeddings)
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(model_name=config.rag.embedding_model)
            except Exception as e:
                print(f"Error setting up local embeddings: {e}")
                # Fallback to OpenAI if available
                if config.ai.openai_api_key:
                    embeddings = OpenAIEmbeddings(openai_api_key=config.ai.openai_api_key)
                else:
                    return
            
            self.vectorstore = Chroma.from_documents(
                texts, 
                embeddings,
                persist_directory=self.vectorstore_path
            )
            
        except Exception as e:
            print(f"Error setting up basic retrieval chain: {e}")
            self.vectorstore = None
    
    def _setup_github_rag(self):
        """Set up RAG for GitHub repository."""
        try:
            from .github_analyzer import GitHubAnalyzer
            github_analyzer = GitHubAnalyzer()
            
            # Download and analyze the repository
            results = github_analyzer.analyze_repository_with_info(self.codebase_path)
            
            if "error" in results:
                print(f"Error analyzing GitHub repository: {results['error']}")
                self.enhanced_rag = None
                return
            
            # Get the downloaded repository path
            repo_path = github_analyzer.temp_dir
            if repo_path and Path(repo_path).exists():
                print(f"✅ GitHub repository downloaded to: {repo_path}")
                
                # Initialize enhanced RAG system with the downloaded repository
                self.enhanced_rag = EnhancedCodeRAGSystem(str(repo_path))
                self.github_data = results.get("github_info", {})
                
                # Set up the retrieval chain and index the repository
                if self._has_api_key():
                    self._initialize_llm()
                    self._setup_enhanced_retrieval_chain()
                    
                    # Ensure the repository is properly indexed
                    if self.enhanced_rag:
                        index_result = self.enhanced_rag.index_codebase()
                        if "error" in index_result:
                            print(f"Error indexing repository: {index_result['error']}")
                        else:
                            print(f"✅ Repository indexed: {index_result.get('documents_indexed', 0)} files, {index_result.get('entities_found', 0)} entities")
            else:
                print("Failed to download GitHub repository")
                self.enhanced_rag = None
                
        except Exception as e:
            print(f"Error setting up GitHub RAG: {e}")
            self.enhanced_rag = None
    
    def _load_codebase_documents(self) -> List[Document]:
        """Load codebase files as documents."""
        documents = []
        
        # Check if this is a GitHub repository analysis
        if hasattr(self, 'github_data') and self.github_data:
            return self._load_github_documents()
        
        # Check if path is a file or directory
        path_obj = Path(self.codebase_path)
        
        if path_obj.is_file():
            # Handle single file
            try:
                loader = TextLoader(str(path_obj))
                docs = loader.load()
                documents.extend(docs)
                # print(f"✅ Loaded single file: {path_obj.name}")
            except Exception as e:
                print(f"Error loading file {path_obj}: {e}")
        else:
            # Handle directory
            supported_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.txt']
            
            for ext in supported_extensions:
                try:
                    loader = DirectoryLoader(
                        str(self.codebase_path),
                        glob=f"**/*{ext}",
                        loader_cls=TextLoader,
                        show_progress=True
                    )
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {ext} files: {e}")
        
        return documents
    
    def _load_github_documents(self) -> List[Document]:
        """Load GitHub repository data as documents for RAG."""
        documents = []
        
        try:
            # Add repository metadata
            repo_info = self.github_data.get('repository_info', {})
            repo_metadata = f"""Repository: {repo_info.get('name', 'Unknown')}
Description: {repo_info.get('description', 'No description')}
Language: {repo_info.get('language', 'Unknown')}
Stars: {repo_info.get('stars', 0)}
URL: {repo_info.get('url', '')}
"""
            
            documents.append(Document(
                page_content=repo_metadata,
                metadata={'source': 'repository_info', 'type': 'metadata'}
            ))
            
            # Add analyzed files content
            for file_result in self.github_data.get('files_analyzed', []):
                file_path = file_result.get('file_path', 'unknown')
                file_content = file_result.get('content', '')
                
                if file_content:
                    documents.append(Document(
                        page_content=f"File: {file_path}\n\n{file_content}",
                        metadata={'source': file_path, 'type': 'code'}
                    ))
            
            # Add analysis results
            analysis_summary = f"""Analysis Summary:
Total Issues: {self.github_data.get('total_issues', 0)}
Issues by Category: {self.github_data.get('issues_by_category', {})}
Issues by Severity: {self.github_data.get('issues_by_severity', {})}
"""
            
            documents.append(Document(
                page_content=analysis_summary,
                metadata={'source': 'analysis_summary', 'type': 'analysis'}
            ))
            
            # print(f"✅ Loaded {len(documents)} GitHub documents for RAG")
            
        except Exception as e:
            print(f"Error loading GitHub documents: {e}")
        
        return documents
    
    def ask_question(self, question: str) -> str:
        """Ask a question about the codebase."""
        if not self._has_api_key():
            provider = config.ai.llm_provider
            return f"Error: Q&A agent not properly initialized. Please check your {provider.upper()} API key."
        
        try:
            if config.ai.llm_provider == "groq":
                return self._ask_groq(question)
            else:
                return self._ask_openai(question)
                
        except Exception as e:
            return f"Error processing question: {str(e)}"
    
    def _ask_groq(self, question: str) -> str:
        """Ask question using Groq API with enhanced context."""
        try:
            # Get enhanced context from vector store
            if hasattr(self, 'vectorstore') and self.vectorstore:
                docs = self.vectorstore.similarity_search(question, k=5)
                context = "\n".join([doc.page_content for doc in docs])
            else:
                context = "No context available."
            
            # Get codebase summary for additional context
            codebase_summary = ""
            if hasattr(self, 'enhanced_rag'):
                codebase_summary = self.enhanced_rag.get_codebase_summary()
            
            # Prepare enhanced prompt
            prompt = f"""You are an expert code analysis assistant with deep understanding of code structure and relationships. Answer the following question about the codebase:

Question: {question}

Codebase Overview:
{codebase_summary}

Relevant Code Context:
{context}

Instructions:
1. Analyze the code structure and relationships
2. Provide specific examples from the code
3. Explain how different parts of the code interact
4. Identify patterns, dependencies, and architectural decisions
5. If asked about specific functions/classes, explain their purpose, parameters, and usage
6. If asked about relationships, trace dependencies and call chains

IMPORTANT: Only answer based on the provided context. If the context doesn't contain the information needed to answer the question, say "I don't have enough information in the provided context to answer this question." Do not make up or hallucinate information."""
            
            # Use LangChain LLM
            response = self.llm.invoke(prompt)
            
            # Extract content from LangChain response
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Add enhanced source information
            if hasattr(self, 'vectorstore') and self.vectorstore:
                docs = self.vectorstore.similarity_search(question, k=3)
                if docs:
                    sources = []
                    for doc in docs:
                        source_file = doc.metadata.get("source", "Unknown")
                        entity_type = doc.metadata.get("type", "code")
                        if entity_type == "entity":
                            entity_name = doc.metadata.get("entity_name", "")
                            sources.append(f"- {Path(source_file).name} ({entity_name})")
                        else:
                            sources.append(f"- {Path(source_file).name}")
                    
                    if sources:
                        answer += f"\n\n**Sources:**\n" + "\n".join(sources)
            
            return answer
            
        except Exception as e:
            return f"Error with Groq API: {str(e)}"
    
    def _ask_openai(self, question: str) -> str:
        """Ask question using OpenAI API."""
        try:
            # Get relevant context from vector store
            if hasattr(self, 'vectorstore') and self.vectorstore:
                docs = self.vectorstore.similarity_search(question, k=3)
                context = "\n".join([doc.page_content for doc in docs])
            else:
                context = "No context available."
            
            # Prepare prompt
            prompt = f"""You are a helpful code analysis assistant. Answer the following question about the codebase:

Question: {question}

Context from codebase:
{context}

Please provide a helpful answer based on the context. If the context doesn't contain enough information, say so."""
            
            # Use LangChain LLM
            response = self.llm.invoke(prompt)
            
            # Extract content from LangChain response
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return answer
            
        except Exception as e:
            return f"Error with OpenAI API: {str(e)}"
    
    def get_codebase_summary(self) -> str:
        """Get a summary of the codebase."""
        return self.ask_question("Please provide a comprehensive summary of this codebase including its main purpose, key components, and overall architecture.")
    
    def find_issues(self, issue_type: str = "all") -> str:
        """Find specific types of issues in the codebase."""
        issue_questions = {
            "security": "What security vulnerabilities can you identify in this codebase?",
            "performance": "What performance bottlenecks do you see in this code?",
            "maintainability": "What maintainability issues are present in this codebase?",
            "testing": "What testing gaps do you observe?",
            "documentation": "What documentation issues do you find?",
            "all": "What are the main quality issues and areas for improvement in this codebase?"
        }
        
        question = issue_questions.get(issue_type, issue_questions["all"])
        return self.ask_question(question)
