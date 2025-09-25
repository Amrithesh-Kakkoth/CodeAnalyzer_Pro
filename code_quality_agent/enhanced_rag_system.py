"""
Enhanced RAG system for comprehensive codebase understanding.
Analyzes code relationships, dependencies, and structure.
"""

import ast
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

from .config import config


@dataclass
class CodeEntity:
    """Represents a code entity (function, class, variable, etc.)."""
    name: str
    type: str  # 'function', 'class', 'variable', 'import', 'module'
    file_path: str
    line_number: int
    content: str
    docstring: Optional[str] = None
    parameters: List[str] = None
    return_type: Optional[str] = None
    dependencies: List[str] = None
    callers: List[str] = None


@dataclass
class CodeRelationship:
    """Represents a relationship between code entities."""
    source: str
    target: str
    relationship_type: str  # 'calls', 'imports', 'inherits', 'uses', 'defines'
    file_path: str
    line_number: int


class CodeStructureAnalyzer:
    """Analyzes code structure and relationships."""
    
    def __init__(self):
        self.entities: Dict[str, CodeEntity] = {}
        self.relationships: List[CodeRelationship] = []
        self.file_dependencies: Dict[str, Set[str]] = {}
        
    def analyze_python_file(self, file_path: str, content: str) -> List[CodeEntity]:
        """Analyze Python file structure."""
        entities = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    entity = CodeEntity(
                        name=node.name,
                        type='function',
                        file_path=file_path,
                        line_number=node.lineno,
                        content=ast.get_source_segment(content, node),
                        docstring=ast.get_docstring(node),
                        parameters=[arg.arg for arg in node.args.args],
                        dependencies=self._extract_function_dependencies(node, content)
                    )
                    entities.append(entity)
                    
                elif isinstance(node, ast.ClassDef):
                    entity = CodeEntity(
                        name=node.name,
                        type='class',
                        file_path=file_path,
                        line_number=node.lineno,
                        content=ast.get_source_segment(content, node),
                        docstring=ast.get_docstring(node),
                        dependencies=self._extract_class_dependencies(node, content)
                    )
                    entities.append(entity)
                    
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        entity = CodeEntity(
                            name=alias.name,
                            type='import',
                            file_path=file_path,
                            line_number=node.lineno,
                            content=f"import {alias.name}",
                            dependencies=[alias.name]
                        )
                        entities.append(entity)
                        
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        entity = CodeEntity(
                            name=alias.name,
                            type='import',
                            file_path=file_path,
                            line_number=node.lineno,
                            content=f"from {node.module} import {alias.name}",
                            dependencies=[f"{node.module}.{alias.name}"]
                        )
                        entities.append(entity)
                        
        except SyntaxError:
            # Handle syntax errors gracefully
            pass
            
        return entities
    
    def analyze_javascript_file(self, file_path: str, content: str) -> List[CodeEntity]:
        """Analyze JavaScript/TypeScript file structure."""
        entities = []
        
        # Function definitions
        function_pattern = r'function\s+(\w+)\s*\([^)]*\)\s*\{'
        for match in re.finditer(function_pattern, content):
            entities.append(CodeEntity(
                name=match.group(1),
                type='function',
                file_path=file_path,
                line_number=content[:match.start()].count('\n') + 1,
                content=match.group(0),
                dependencies=self._extract_js_dependencies(content, match.start())
            ))
        
        # Arrow functions
        arrow_pattern = r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>'
        for match in re.finditer(arrow_pattern, content):
            entities.append(CodeEntity(
                name=match.group(1),
                type='function',
                file_path=file_path,
                line_number=content[:match.start()].count('\n') + 1,
                content=match.group(0),
                dependencies=self._extract_js_dependencies(content, match.start())
            ))
        
        # Class definitions
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{'
        for match in re.finditer(class_pattern, content):
            entities.append(CodeEntity(
                name=match.group(1),
                type='class',
                file_path=file_path,
                line_number=content[:match.start()].count('\n') + 1,
                content=match.group(0),
                dependencies=self._extract_js_dependencies(content, match.start())
            ))
        
        # Import statements
        import_pattern = r'import\s+(?:\{[^}]*\}|\w+)\s+from\s+[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(import_pattern, content):
            entities.append(CodeEntity(
                name=match.group(1),
                type='import',
                file_path=file_path,
                line_number=content[:match.start()].count('\n') + 1,
                content=match.group(0),
                dependencies=[match.group(1)]
            ))
        
        return entities
    
    def _extract_function_dependencies(self, node: ast.FunctionDef, content: str) -> List[str]:
        """Extract dependencies from a function."""
        dependencies = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                dependencies.append(child.id)
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.append(child.func.attr)
        
        return list(set(dependencies))
    
    def _extract_class_dependencies(self, node: ast.ClassDef, content: str) -> List[str]:
        """Extract dependencies from a class."""
        dependencies = []
        
        # Base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                dependencies.append(base.id)
        
        # Method dependencies
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                dependencies.append(child.id)
        
        return list(set(dependencies))
    
    def _extract_js_dependencies(self, content: str, start_pos: int) -> List[str]:
        """Extract dependencies from JavaScript code."""
        dependencies = []
        
        # Find the end of the function/class
        brace_count = 0
        in_function = False
        for i, char in enumerate(content[start_pos:], start_pos):
            if char == '{':
                brace_count += 1
                in_function = True
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and in_function:
                    end_pos = i
                    break
        else:
            end_pos = len(content)
        
        # Extract function calls and variable references
        function_content = content[start_pos:end_pos]
        
        # Function calls
        call_pattern = r'(\w+)\s*\('
        for match in re.finditer(call_pattern, function_content):
            dependencies.append(match.group(1))
        
        # Variable references
        var_pattern = r'\b(\w+)\b'
        for match in re.finditer(var_pattern, function_content):
            if match.group(1) not in ['var', 'let', 'const', 'function', 'if', 'else', 'for', 'while', 'return']:
                dependencies.append(match.group(1))
        
        return list(set(dependencies))
    
    def build_relationships(self, entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Build relationships between entities."""
        relationships = []
        
        for entity in entities:
            if entity.dependencies:
                for dep in entity.dependencies:
                    # Find the dependency entity
                    dep_entity = self._find_entity_by_name(dep, entities)
                    if dep_entity:
                        relationship = CodeRelationship(
                            source=entity.name,
                            target=dep_entity.name,
                            relationship_type='uses',
                            file_path=entity.file_path,
                            line_number=entity.line_number
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _find_entity_by_name(self, name: str, entities: List[CodeEntity]) -> Optional[CodeEntity]:
        """Find entity by name."""
        for entity in entities:
            if entity.name == name:
                return entity
        return None


class EnhancedCodeRAGSystem:
    """Enhanced RAG system with code structure understanding."""
    
    def __init__(self, codebase_path: str):
        """Initialize enhanced RAG system."""
        self.codebase_path = Path(codebase_path)
        self.vectorstore = None
        self.embeddings = None
        self.text_splitter = None
        self.structure_analyzer = CodeStructureAnalyzer()
        self.code_entities: Dict[str, List[CodeEntity]] = {}
        self.code_relationships: List[CodeRelationship] = []
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_text_splitter()
    
    def _initialize_embeddings(self):
        """Initialize embeddings model."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=config.rag.embedding_model)
        except ImportError:
            try:
                import subprocess
                subprocess.check_call(["pip", "install", "sentence-transformers"])
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(model_name=config.rag.embedding_model)
            except Exception:
                # Fallback to OpenAI if available
                try:
                    from langchain_openai import OpenAIEmbeddings
                    import os
                    if os.getenv("OPENAI_API_KEY"):
                        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
                    else:
                        self.embeddings = None
                except Exception:
                    self.embeddings = None
        except Exception as e:
            print(f"Warning: Failed to initialize embeddings: {e}")
            self.embeddings = None
    
    def _initialize_text_splitter(self):
        """Initialize text splitter for code."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap
        )
    
    def index_codebase(self) -> Dict[str, Any]:
        """Index the entire codebase with structure analysis."""
        try:
            # Analyze code structure
            self._analyze_codebase_structure()
            
            # Load documents with enhanced metadata
            documents = self._load_enhanced_documents()
            
            if not documents:
                return {"error": "No documents found to index"}
            
            # Filter complex metadata to avoid vector store errors
            filtered_documents = []
            for doc in documents:
                try:
                    # Create safe metadata that preserves important information
                    filtered_metadata = self._create_safe_metadata(doc.metadata)
                    filtered_doc = Document(
                        page_content=doc.page_content,
                        metadata=filtered_metadata
                    )
                    filtered_documents.append(filtered_doc)
                except Exception as e:
                    print(f"Warning: Error filtering metadata for document: {e}")
                    # Fallback: create document with basic metadata
                    filtered_doc = Document(
                        page_content=doc.page_content,
                        metadata={
                            'source': doc.metadata.get('source', 'unknown'),
                            'filename': doc.metadata.get('filename', 'unknown'),
                            'type': doc.metadata.get('type', 'unknown'),
                            'entity_name': doc.metadata.get('entity_name', ''),
                            'line_number': str(doc.metadata.get('line_number', '')),
                            'language': doc.metadata.get('language', 'unknown')
                        }
                    )
                    filtered_documents.append(filtered_doc)
            
            # Split into chunks (text_splitter expects Document objects)
            texts = self.text_splitter.split_documents(filtered_documents)
            
            # Create vector store
            if self.embeddings:
                try:
                    self.vectorstore = Chroma.from_documents(
                        texts,
                        self.embeddings,
                        persist_directory=f"{config.rag.vector_db_path}_enhanced"
                    )
                    
                    return {
                        "success": True,
                        "documents_indexed": len(documents),
                        "chunks_created": len(texts),
                        "entities_found": sum(len(entities) for entities in self.code_entities.values()),
                        "relationships_found": len(self.code_relationships)
                    }
                except Exception as e:
                    return {"error": f"Failed to create vector store: {str(e)}"}
            else:
                return {"error": "No embedding model available - enhanced mode requires API keys"}
                
        except Exception as e:
            return {"error": f"Error indexing codebase: {str(e)}"}
    
    def _analyze_codebase_structure(self):
        """Analyze the structure of the entire codebase."""
        self.code_entities = {}
        self.code_relationships = []
        
        if self.codebase_path.is_file():
            self._analyze_file(self.codebase_path)
        else:
            # Analyze all supported files in directory
            supported_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx']
            for ext in supported_extensions:
                for file_path in self.codebase_path.rglob(f'*{ext}'):
                    self._analyze_file(file_path)
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single file."""
        try:
            print(f"🔍 Analyzing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_path.suffix == '.py':
                entities = self.structure_analyzer.analyze_python_file(str(file_path), content)
                print(f"  📊 Found {len(entities)} Python entities")
            elif file_path.suffix in ['.js', '.ts', '.jsx', '.tsx']:
                entities = self.structure_analyzer.analyze_javascript_file(str(file_path), content)
                print(f"  📊 Found {len(entities)} JavaScript/TypeScript entities")
            else:
                print(f"  ⚠️ Unsupported file type: {file_path.suffix}")
                return
            
            self.code_entities[str(file_path)] = entities
            
            # Build relationships for this file
            relationships = self.structure_analyzer.build_relationships(entities)
            self.code_relationships.extend(relationships)
            print(f"  🔗 Added {len(relationships)} relationships")
            
        except Exception as e:
            print(f"  ❌ Error analyzing {file_path}: {e}")
    
    def debug_vector_store(self) -> Dict[str, Any]:
        """Debug information about the vector store."""
        if not self.vectorstore:
            return {"error": "No vector store available"}
        
        try:
            all_docs = self.vectorstore.get()
            debug_info = {
                "total_documents": len(all_docs.get('documents', [])),
                "files": {},
                "entity_types": {},
                "languages": {}
            }
            
            if 'documents' in all_docs and 'metadatas' in all_docs:
                for i, metadata in enumerate(all_docs['metadatas']):
                    filename = metadata.get('filename', 'unknown')
                    entity_type = metadata.get('type', 'unknown')
                    language = metadata.get('language', 'unknown')
                    
                    # Count by file
                    if filename not in debug_info['files']:
                        debug_info['files'][filename] = 0
                    debug_info['files'][filename] += 1
                    
                    # Count by entity type
                    if entity_type not in debug_info['entity_types']:
                        debug_info['entity_types'][entity_type] = 0
                    debug_info['entity_types'][entity_type] += 1
                    
                    # Count by language
                    if language not in debug_info['languages']:
                        debug_info['languages'][language] = 0
                    debug_info['languages'][language] += 1
            
            return debug_info
            
        except Exception as e:
            return {"error": f"Debug failed: {e}"}
    
    def search_debug(self, query: str) -> Dict[str, Any]:
        """Debug search results for a query."""
        debug_info = {
            "query": query,
            "filename_detected": None,
            "file_search_results": [],
            "similarity_search_results": [],
            "total_results": 0
        }
        
        # Check filename detection
        filename_match = self._extract_filename_from_query(query)
        debug_info["filename_detected"] = filename_match
        
        if filename_match:
            file_docs = self.search_by_filename(filename_match)
            debug_info["file_search_results"] = [
                {
                    "filename": doc.metadata.get('filename', 'unknown'),
                    "type": doc.metadata.get('type', 'unknown'),
                    "entity_name": doc.metadata.get('entity_name', ''),
                    "line_number": doc.metadata.get('line_number', ''),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                for doc in file_docs
            ]
        
        # Check similarity search
        try:
            docs = self.vectorstore.similarity_search(query, k=5)
            debug_info["similarity_search_results"] = [
                {
                    "filename": doc.metadata.get('filename', 'unknown'),
                    "type": doc.metadata.get('type', 'unknown'),
                    "entity_name": doc.metadata.get('entity_name', ''),
                    "line_number": doc.metadata.get('line_number', ''),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                for doc in docs
            ]
        except Exception as e:
            debug_info["similarity_search_error"] = str(e)
        
        debug_info["total_results"] = len(debug_info["file_search_results"]) + len(debug_info["similarity_search_results"])
        
        return debug_info
    
    def _load_enhanced_documents(self) -> List[Document]:
        """Load documents with enhanced metadata."""
        documents = []
        
        for file_path, entities in self.code_entities.items():
            # Create document for the entire file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get file info
            file_info = Path(file_path)
            file_extension = file_info.suffix
            file_name = file_info.name
            
            # Enhanced metadata with comprehensive information (filtered for vector store)
            metadata = {
                'source': file_path,
                'filename': file_name,
                'file_extension': file_extension,
                'type': 'code_file',
                'total_entities': str(len(entities)),
                'entity_types': ', '.join(set(e.type for e in entities)),  # Convert list to string
                'entities': json.dumps([{
                    'name': e.name,
                    'type': e.type,
                    'line_number': e.line_number,
                    'docstring': e.docstring,
                    'parameters': e.parameters,
                    'dependencies': e.dependencies
                } for e in entities]),
                'relationships': json.dumps([{
                    'source': r.source,
                    'target': r.target,
                    'type': r.relationship_type,
                    'line_number': r.line_number
                } for r in self.code_relationships if r.file_path == file_path]),
                'ast_parsed': 'True',
                'language': self._detect_language(file_extension),
                'code_content': content  # Include the actual code content
            }
            
            # Create comprehensive file document
            file_doc_content = f"""
File: {file_name}
Language: {metadata['language']}
Extension: {file_extension}
Total Entities: {len(entities)}
Entity Types: {', '.join(set(e.type for e in entities))}

Code Content:
{content}

AST Parsed Entities:
{self._format_entities_for_document(entities)}
            """.strip()
            
            documents.append(Document(page_content=file_doc_content, metadata=metadata))
            
            # Create individual entity documents with enhanced metadata
            for entity in entities:
                entity_doc_content = f"""
Entity: {entity.name}
Type: {entity.type}
File: {file_name}
Line: {entity.line_number}
Language: {metadata['language']}
AST Parsed: True

Code Content:
{entity.content}

Docstring: {entity.docstring or 'No docstring'}
Parameters: {entity.parameters or 'No parameters'}
Dependencies: {entity.dependencies or 'No dependencies'}

Context: This {entity.type} is defined in {file_name} at line {entity.line_number}.
                """.strip()
                
                entity_doc = Document(
                    page_content=entity_doc_content,
                    metadata={
                        'source': file_path,
                        'filename': file_name,
                        'file_extension': file_extension,
                        'type': 'entity',
                        'entity_name': entity.name,
                        'entity_type': entity.type,
                        'line_number': str(entity.line_number),
                        'language': metadata['language'],
                        'ast_parsed': 'True',
                        'has_docstring': str(bool(entity.docstring)),
                        'has_parameters': str(bool(entity.parameters)),
                        'has_dependencies': str(bool(entity.dependencies)),
                        'code_content': entity.content  # Include the actual code content
                    }
                )
                documents.append(entity_doc)
        
        return documents
    
    def _detect_language(self, file_extension: str) -> str:
        """Detect programming language from file extension."""
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'JavaScript',
            '.tsx': 'TypeScript',
            '.md': 'Markdown',
            '.txt': 'Text',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML'
        }
        return language_map.get(file_extension.lower(), 'Unknown')
    
    def _format_entities_for_document(self, entities: List[CodeEntity]) -> str:
        """Format entities for document content."""
        if not entities:
            return "No entities found"
        
        formatted = []
        for entity in entities:
            formatted.append(f"- {entity.name} ({entity.type}) at line {entity.line_number}")
            if entity.docstring:
                formatted.append(f"  Docstring: {entity.docstring[:100]}...")
            if entity.parameters:
                formatted.append(f"  Parameters: {', '.join(entity.parameters)}")
        
        return '\n'.join(formatted)
    
    def search_similar(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar code with enhanced context."""
        if not self.vectorstore:
            return []
        
        # Check if query mentions a specific file
        filename_match = self._extract_filename_from_query(query)
        if filename_match:
            print(f"🔍 Detected filename query: {filename_match}")
            # First try to find the specific file
            file_docs = self.search_by_filename(filename_match)
            if file_docs:
                print(f"✅ Found {len(file_docs)} documents for {filename_match}")
                # Return only the most relevant documents to avoid token limits
                return file_docs[:min(k, 3)]
            else:
                print(f"⚠️ File {filename_match} not found, falling back to similarity search")
        
        # Regular similarity search
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            print(f"🔍 Found {len(docs)} similar documents for query: {query[:50]}...")
            
            # Enhance results with relationship context and metadata (truncated)
            enhanced_docs = []
            for doc in docs:
                enhanced_content = self._enhance_document_context(doc, query)
                # Truncate content to avoid token limits
                if len(enhanced_content) > 1000:
                    enhanced_content = enhanced_content[:1000] + "..."
                
                enhanced_docs.append(Document(
                    page_content=enhanced_content,
                    metadata=doc.metadata
                ))
            
            # Add related entities based on query with enhanced metadata (limited)
            related_entities = self._find_related_entities(query)
            for entity in related_entities[:2]:  # Limit to 2 related entities
                file_info = Path(entity.file_path)
                # Truncate entity content
                entity_content = entity.content[:500] + "..." if len(entity.content) > 500 else entity.content
                
                entity_doc_content = f"""
Entity: {entity.name}
Type: {entity.type}
File: {file_info.name}
Line: {entity.line_number}

Code Content:
{entity_content}

Docstring: {entity.docstring or 'No docstring'}
Parameters: {entity.parameters or 'No parameters'}
                """.strip()
                
                entity_doc = Document(
                    page_content=entity_doc_content,
                    metadata={
                        'source': entity.file_path,
                        'filename': file_info.name,
                        'file_extension': file_info.suffix,
                        'type': 'related_entity',
                        'entity_name': entity.name,
                        'entity_type': entity.type,
                        'line_number': str(entity.line_number),
                        'language': self._detect_language(file_info.suffix),
                        'ast_parsed': 'True',
                        'has_docstring': str(bool(entity.docstring)),
                        'has_parameters': str(bool(entity.parameters)),
                        'has_dependencies': str(bool(entity.dependencies)),
                        'code_content': entity_content
                    }
                )
                enhanced_docs.append(entity_doc)
            
            return enhanced_docs[:k]
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def _extract_filename_from_query(self, query: str) -> Optional[str]:
        """Extract filename from query if mentioned."""
        import re
        
        # Look for common file patterns
        patterns = [
            r'(\w+\.py)',      # test.py
            r'(\w+\.js)',      # test.js
            r'(\w+\.ts)',      # test.ts
            r'(\w+\.jsx)',     # test.jsx
            r'(\w+\.tsx)',     # test.tsx
            r'file\s+(\w+\.\w+)',  # file test.py
            r'in\s+(\w+\.\w+)',    # in test.py
            r'from\s+(\w+\.\w+)',  # from test.py
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _enhance_document_context(self, doc: Document, query: str) -> str:
        """Enhance document with relationship context."""
        content = doc.page_content
        metadata = doc.metadata
        
        # Add comprehensive metadata information
        if 'filename' in metadata:
            content += f"\n\n**File Information:**\n"
            content += f"- Filename: {metadata['filename']}\n"
            content += f"- Language: {metadata.get('language', 'Unknown')}\n"
            content += f"- Extension: {metadata.get('file_extension', 'Unknown')}\n"
            content += f"- AST Parsed: {metadata.get('ast_parsed', False)}\n"
            content += f"- Total Entities: {metadata.get('total_entities', 0)}\n"
            content += f"- Entity Types: {', '.join(metadata.get('entity_types', []))}\n"
        
        # Add entity information
        if 'entities' in metadata:
            try:
                entities = json.loads(metadata['entities'])
                if entities:
                    content += "\n\n**AST Parsed Entities:**\n"
                    for entity in entities:
                        content += f"- {entity['name']} ({entity['type']}) at line {entity['line_number']}\n"
                        if entity.get('docstring'):
                            content += f"  Docstring: {entity['docstring'][:100]}...\n"
                        if entity.get('parameters'):
                            content += f"  Parameters: {', '.join(entity['parameters'])}\n"
                        if entity.get('dependencies'):
                            content += f"  Dependencies: {', '.join(entity['dependencies'][:5])}\n"
            except:
                pass
        
        # Add relationship information
        if 'relationships' in metadata:
            try:
                relationships = json.loads(metadata['relationships'])
                if relationships:
                    content += "\n\n**Code Relationships:**\n"
                    for rel in relationships:
                        content += f"- {rel['source']} {rel['type']} {rel['target']} (line {rel['line_number']})\n"
            except:
                pass
        
        return content
    
    def get_codebase_summary(self) -> str:
        """Get comprehensive codebase summary."""
        summary = f"Codebase Summary for {self.codebase_path.name}\n\n"
        
        # File statistics
        total_files = len(self.code_entities)
        total_entities = sum(len(entities) for entities in self.code_entities.values())
        total_relationships = len(self.code_relationships)
        
        summary += f"**Files Analyzed:** {total_files}\n"
        summary += f"**Code Entities:** {total_entities}\n"
        summary += f"**Relationships:** {total_relationships}\n\n"
        
        # Entity breakdown
        entity_types = {}
        for entities in self.code_entities.values():
            for entity in entities:
                entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        
        summary += "**Entity Types:**\n"
        for entity_type, count in entity_types.items():
            summary += f"- {entity_type}: {count}\n"
        
        # Top entities by dependencies
        entity_deps = {}
        for entities in self.code_entities.values():
            for entity in entities:
                if entity.dependencies:
                    entity_deps[entity.name] = len(entity.dependencies)
        
        if entity_deps:
            summary += "\n**Most Complex Entities (by dependencies):**\n"
            sorted_entities = sorted(entity_deps.items(), key=lambda x: x[1], reverse=True)
            for name, deps in sorted_entities[:5]:
                summary += f"- {name}: {deps} dependencies\n"
        
        return summary
    
    def find_entity_by_name(self, name: str) -> List[CodeEntity]:
        """Find entities by name across the codebase."""
        results = []
        for entities in self.code_entities.values():
            for entity in entities:
                if name.lower() in entity.name.lower():
                    results.append(entity)
        return results
    
    def get_entity_relationships(self, entity_name: str) -> List[CodeRelationship]:
        """Get all relationships for a specific entity."""
        relationships = []
        for rel in self.code_relationships:
            if rel.source == entity_name or rel.target == entity_name:
                relationships.append(rel)
        return relationships
    
    def _create_safe_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Create safe metadata that preserves important information."""
        safe_metadata = {}
        
        # Essential fields that must be preserved
        essential_fields = [
            'source', 'filename', 'file_extension', 'type', 'entity_name', 
            'entity_type', 'line_number', 'language', 'ast_parsed',
            'has_docstring', 'has_parameters', 'has_dependencies'
        ]
        
        for field in essential_fields:
            value = metadata.get(field)
            if value is not None:
                # Convert to string and limit length
                safe_metadata[field] = str(value)[:500]  # Limit to 500 chars
        
        # Add code content if available (truncated)
        if 'code_content' in metadata:
            content = metadata['code_content']
            if content:
                safe_metadata['code_content'] = str(content)[:1000]  # Limit to 1000 chars
        
        return safe_metadata
    
    def search_by_filename(self, filename: str) -> List[Document]:
        """Search for documents by exact filename match."""
        if not self.vectorstore:
            return []
        
        try:
            # Get all documents from the vector store
            all_docs = self.vectorstore.get()
            matching_docs = []
            
            if all_docs and 'documents' in all_docs:
                for i, doc_content in enumerate(all_docs['documents']):
                    metadata = all_docs['metadatas'][i] if 'metadatas' in all_docs else {}
                    if metadata.get('filename') == filename:
                        matching_docs.append(Document(
                            page_content=doc_content,
                            metadata=metadata
                        ))
            
            # Limit results to avoid token limits - prioritize entities over raw code
            if len(matching_docs) > 10:
                # Sort by type priority: entities first, then code
                entity_docs = [doc for doc in matching_docs if doc.metadata.get('type') == 'entity']
                code_docs = [doc for doc in matching_docs if doc.metadata.get('type') != 'entity']
                
                # Take up to 8 entities and 2 code chunks
                limited_docs = entity_docs[:8] + code_docs[:2]
                print(f"📊 Limited {len(matching_docs)} documents to {len(limited_docs)} to avoid token limits")
                return limited_docs
            
            return matching_docs
        except Exception as e:
            print(f"Error searching by filename: {e}")
            return []
    
    def search_by_file_extension(self, extension: str) -> List[Document]:
        """Search for documents by file extension."""
        if not self.vectorstore:
            return []
        
        try:
            all_docs = self.vectorstore.get()
            matching_docs = []
            
            if all_docs and 'documents' in all_docs:
                for i, doc_content in enumerate(all_docs['documents']):
                    metadata = all_docs['metadatas'][i] if 'metadatas' in all_docs else {}
                    if metadata.get('file_extension') == extension:
                        matching_docs.append(Document(
                            page_content=doc_content,
                            metadata=metadata
                        ))
            
            return matching_docs
        except Exception as e:
            print(f"Error searching by file extension: {e}")
            return []
    
    def get_file_content(self, filename: str) -> str:
        """Get the complete content of a specific file."""
        docs = self.search_by_filename(filename)
        if not docs:
            return f"File '{filename}' not found in the indexed codebase."
        
        # Combine all documents for this file, but limit total content
        content_parts = []
        total_length = 0
        max_content_length = 3000  # Limit to 3000 characters
        
        for doc in docs:
            doc_content = doc.page_content
            if total_length + len(doc_content) > max_content_length:
                # Truncate the last document
                remaining_space = max_content_length - total_length
                if remaining_space > 100:  # Only add if there's meaningful space
                    content_parts.append(doc_content[:remaining_space] + "...")
                break
            
            content_parts.append(doc_content)
            total_length += len(doc_content)
        
        return "\n\n".join(content_parts)
    
    def _find_related_entities(self, query: str) -> List[CodeEntity]:
        """Find entities related to the query."""
        related_entities = []
        query_lower = query.lower()
        
        # Find entities by name match
        for entities in self.code_entities.values():
            for entity in entities:
                if query_lower in entity.name.lower() or entity.name.lower() in query_lower:
                    related_entities.append(entity)
        
        # Find entities by type match
        if 'function' in query_lower:
            for entities in self.code_entities.values():
                for entity in entities:
                    if entity.type == 'function':
                        related_entities.append(entity)
        elif 'class' in query_lower:
            for entities in self.code_entities.values():
                for entity in entities:
                    if entity.type == 'class':
                        related_entities.append(entity)
        
        # Remove duplicates
        seen = set()
        unique_entities = []
        for entity in related_entities:
            if entity.name not in seen:
                seen.add(entity.name)
                unique_entities.append(entity)
        
        return unique_entities[:3]  # Limit to 3 related entities
