"""
RAG (Retrieval-Augmented Generation) system for handling large codebases.
"""

from pathlib import Path
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document

from .config import config


class CodeRAGSystem:
    """RAG system for code analysis and retrieval."""
    
    def __init__(self, codebase_path: str):
        """Initialize RAG system with a codebase."""
        self.codebase_path = Path(codebase_path)
        self.vectorstore = None
        self.embeddings = None
        self.text_splitter = None
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_text_splitter()
    
    def _initialize_embeddings(self):
        """Initialize embeddings model."""
        # Use local embeddings by default (free and works with Groq)
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.rag.embedding_model)
            self.embeddings = model
            # print(f"✅ Using local embeddings: {config.rag.embedding_model}")
        except ImportError:
            # print("Warning: sentence-transformers not available. Installing...")
            try:
                import subprocess
                subprocess.check_call(["pip", "install", "sentence-transformers"])
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(config.rag.embedding_model)
                self.embeddings = model
                # print(f"✅ Installed and using local embeddings: {config.rag.embedding_model}")
            except Exception as e:
                # print(f"❌ Failed to install sentence-transformers: {e}. RAG functionality will be limited.")
                self.embeddings = None
    
    def _initialize_text_splitter(self):
        """Initialize text splitter for code."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap
        )
    
    def index_codebase(self) -> Dict[str, Any]:
        """Index the entire codebase for retrieval."""
        try:
            # Load documents
            documents = self._load_codebase_documents()
            
            if not documents:
                return {"error": "No documents found to index"}
            
            # Split into chunks
            texts = self.text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            if not self.embeddings:
                return {"error": "No embedding model available"}
            
            # Convert to LangChain compatible embeddings
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name=config.rag.embedding_model)
            
            self.vectorstore = Chroma.from_documents(
                texts,
                embeddings,
                persist_directory=config.rag.vector_db_path
            )
            
            return {
                "success": True,
                "documents_processed": len(documents),
                "chunks_created": len(texts),
                "vector_store_path": config.rag.vector_db_path
            }
            
        except Exception as e:
            return {"error": f"Failed to index codebase: {str(e)}"}
    
    def _load_codebase_documents(self) -> List[Document]:
        """Load codebase files as documents."""
        documents = []
        
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
            supported_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.txt', '.json', '.yaml', '.yml']
            
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
    
    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents."""
        if not self.vectorstore:
            return []
        
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error searching vectorstore: {e}")
            return []
    
    def get_codebase_summary(self) -> str:
        """Get a summary of the indexed codebase."""
        if not self.vectorstore:
            return "No codebase indexed yet."
        
        try:
            # Get a sample of documents
            sample_docs = self.vectorstore.similarity_search("", k=10)
            
            if not sample_docs:
                return "No documents found in the codebase."
            
            # Create a basic summary
            file_types = {}
            total_chunks = len(sample_docs)
            
            for doc in sample_docs:
                source = doc.metadata.get("source", "unknown")
                ext = Path(source).suffix
                file_types[ext] = file_types.get(ext, 0) + 1
            
            summary = f"Codebase Summary:\n"
            summary += f"- Total indexed chunks: {total_chunks}\n"
            summary += f"- File types found: {', '.join(file_types.keys())}\n"
            
            return summary
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"

