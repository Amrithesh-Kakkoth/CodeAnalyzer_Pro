"""
Core code analysis engine for the Code Quality Intelligence Agent.
"""

import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .config import config
from .models import CodeIssue, Severity, LanguageDetector


class PythonAnalyzer:
    """Analyzes Python code for quality issues."""
    
    def __init__(self):
        self.issues: List[CodeIssue] = []
    
    def analyze_file(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze a Python file for quality issues."""
        self.issues = []
        
        try:
            tree = ast.parse(content, filename=file_path)
            self._analyze_ast(tree, file_path)
            self._analyze_security_patterns(content, file_path)
            self._analyze_complexity(content, file_path)
            self._analyze_documentation(content, file_path)
        except SyntaxError as e:
            self.issues.append(CodeIssue(
                message=f"Syntax error: {e.msg}",
                severity=Severity.CRITICAL,
                file=file_path,
                line=e.lineno,
                category="syntax"
            ))
        except Exception as e:
            self.issues.append(CodeIssue(
                message=f"Analysis error: {str(e)}",
                severity=Severity.HIGH,
                file=file_path,
                category="analysis"
            ))
        
        return self.issues
    
    def _analyze_ast(self, tree: ast.AST, file_path: str):
        """Analyze AST for structural issues."""
        for node in ast.walk(tree):
            # Check for long functions
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 50:
                    self.issues.append(CodeIssue(
                        message=f"Function '{node.name}' is too long ({len(node.body)} lines)",
                        severity=Severity.MEDIUM,
                        file=file_path,
                        line=node.lineno,
                        category="complexity",
                        suggestion="Consider breaking this function into smaller functions"
                    ))
                
                # Check for too many parameters
                if len(node.args.args) > 7:
                    self.issues.append(CodeIssue(
                        message=f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                        severity=Severity.MEDIUM,
                        file=file_path,
                        line=node.lineno,
                        category="complexity",
                        suggestion="Consider using a data class or dictionary for parameters"
                    ))
            
            # Check for nested loops
            elif isinstance(node, ast.For):
                if self._get_nesting_level(node) > 3:
                    self.issues.append(CodeIssue(
                        message="Deeply nested loop detected",
                        severity=Severity.MEDIUM,
                        file=file_path,
                        line=node.lineno,
                        category="complexity",
                        suggestion="Consider extracting nested logic into separate functions"
                    ))
    
    def _analyze_security_patterns(self, content: str, file_path: str):
        """Analyze content for security vulnerabilities."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for hardcoded passwords
            if re.search(r'password\s*=\s*["\'][^"\']+["\']', line, re.IGNORECASE):
                self.issues.append(CodeIssue(
                    message="Hardcoded password detected",
                    severity=Severity.HIGH,
                    file=file_path,
                    line=i,
                    category="security",
                    suggestion="Use environment variables or secure configuration management"
                ))
            
            # Check for SQL injection patterns
            if re.search(r'execute\s*\(\s*["\'].*%s.*["\']', line):
                self.issues.append(CodeIssue(
                    message="Potential SQL injection vulnerability",
                    severity=Severity.HIGH,
                    file=file_path,
                    line=i,
                    category="security",
                    suggestion="Use parameterized queries or ORM methods"
                ))
            
            # Check for eval usage
            if 'eval(' in line:
                self.issues.append(CodeIssue(
                    message="Use of eval() detected",
                    severity=Severity.CRITICAL,
                    file=file_path,
                    line=i,
                    category="security",
                    suggestion="Avoid eval() as it can execute arbitrary code"
                ))
    
    def _analyze_complexity(self, content: str, file_path: str):
        """Analyze code complexity."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for long lines
            if len(line) > 120:
                self.issues.append(CodeIssue(
                    message=f"Line too long ({len(line)} characters)",
                    severity=Severity.LOW,
                    file=file_path,
                    line=i,
                    category="style",
                    suggestion="Break long lines for better readability"
                ))
            
            # Check for multiple statements on one line
            if ';' in line and not line.strip().startswith('#'):
                self.issues.append(CodeIssue(
                    message="Multiple statements on one line",
                    severity=Severity.LOW,
                    file=file_path,
                    line=i,
                    category="style",
                    suggestion="Use separate lines for each statement"
                ))
    
    def _analyze_documentation(self, content: str, file_path: str):
        """Analyze documentation coverage."""
        lines = content.split('\n')
        functions = []
        classes = []
        
        # Find functions and classes
        for i, line in enumerate(lines, 1):
            if re.match(r'^\s*def\s+\w+', line):
                functions.append((i, line.strip()))
            elif re.match(r'^\s*class\s+\w+', line):
                classes.append((i, line.strip()))
        
        # Check for missing docstrings
        for line_num, func_line in functions:
            func_name = re.search(r'def\s+(\w+)', func_line).group(1)
            if not self._has_docstring(lines, line_num):
                self.issues.append(CodeIssue(
                    message=f"Function '{func_name}' missing docstring",
                    severity=Severity.LOW,
                    file=file_path,
                    line=line_num,
                    category="documentation",
                    suggestion="Add a docstring describing the function's purpose"
                ))
    
    def _has_docstring(self, lines: List[str], start_line: int) -> bool:
        """Check if a function has a docstring."""
        if start_line >= len(lines):
            return False
        
        # Look for docstring in the next few lines
        for i in range(start_line, min(start_line + 5, len(lines))):
            line = lines[i].strip()
            if '"""' in line or "'''" in line:
                return True
            if line and not line.startswith('#'):
                break
        
        return False
    
    def _get_nesting_level(self, node: ast.AST) -> int:
        """Calculate nesting level of a node."""
        level = 0
        current = node
        while hasattr(current, 'parent'):
            current = current.parent
            level += 1
        return level


class JavaScriptAnalyzer:
    """Analyzes JavaScript/TypeScript code for quality issues."""
    
    def __init__(self):
        self.issues: List[CodeIssue] = []
    
    def analyze_file(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze a JavaScript/TypeScript file for quality issues."""
        self.issues = []
        
        self._analyze_security_patterns(content, file_path)
        self._analyze_complexity(content, file_path)
        self._analyze_modern_patterns(content, file_path)
        
        return self.issues
    
    def _analyze_security_patterns(self, content: str, file_path: str):
        """Analyze content for security vulnerabilities."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for eval usage
            if 'eval(' in line:
                self.issues.append(CodeIssue(
                    message="Use of eval() detected",
                    severity=Severity.CRITICAL,
                    file=file_path,
                    line=i,
                    category="security",
                    suggestion="Avoid eval() as it can execute arbitrary code"
                ))
            
            # Check for innerHTML usage
            if '.innerHTML' in line and '=' in line:
                self.issues.append(CodeIssue(
                    message="Direct innerHTML assignment detected",
                    severity=Severity.HIGH,
                    file=file_path,
                    line=i,
                    category="security",
                    suggestion="Use textContent or proper DOM manipulation to prevent XSS"
                ))
            
            # Check for console.log in production code
            if 'console.log(' in line:
                self.issues.append(CodeIssue(
                    message="Console.log statement found",
                    severity=Severity.LOW,
                    file=file_path,
                    line=i,
                    category="style",
                    suggestion="Remove or replace with proper logging in production"
                ))
    
    def _analyze_complexity(self, content: str, file_path: str):
        """Analyze code complexity."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for long lines
            if len(line) > 120:
                self.issues.append(CodeIssue(
                    message=f"Line too long ({len(line)} characters)",
                    severity=Severity.LOW,
                    file=file_path,
                    line=i,
                    category="style",
                    suggestion="Break long lines for better readability"
                ))
            
            # Check for deep nesting
            indent_level = len(line) - len(line.lstrip())
            if indent_level > 20:  # More than 5 levels of indentation
                self.issues.append(CodeIssue(
                    message="Deeply nested code detected",
                    severity=Severity.MEDIUM,
                    file=file_path,
                    line=i,
                    category="complexity",
                    suggestion="Consider extracting nested logic into separate functions"
                ))
    
    def _analyze_modern_patterns(self, content: str, file_path: str):
        """Analyze for modern JavaScript/TypeScript patterns."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for var usage (prefer let/const)
            if re.search(r'\bvar\s+\w+', line):
                self.issues.append(CodeIssue(
                    message="Use of 'var' detected",
                    severity=Severity.LOW,
                    file=file_path,
                    line=i,
                    category="style",
                    suggestion="Use 'let' or 'const' instead of 'var'"
                ))
            
            # Check for == instead of ===
            if '==' in line and '===' not in line and '!=' not in line:
                self.issues.append(CodeIssue(
                    message="Use of loose equality (==) detected",
                    severity=Severity.LOW,
                    file=file_path,
                    line=i,
                    category="style",
                    suggestion="Use strict equality (===) instead"
                ))


class CodeAnalyzer:
    """Main code analyzer that coordinates different analysis engines."""
    
    def __init__(self, enhanced_mode: bool = True):
        self.python_analyzer = PythonAnalyzer()
        self.javascript_analyzer = JavaScriptAnalyzer()
        self.duplication_detector = DuplicationDetector()
        self.enhanced_mode = enhanced_mode
        
        if enhanced_mode:
            from .enhanced_analyzer import EnhancedCodeAnalyzer
            from .severity_scorer import SeverityScorer
            self.enhanced_analyzer = EnhancedCodeAnalyzer()
            self.severity_scorer = SeverityScorer()
    
    def analyze_path(self, path: str) -> Dict[str, Any]:
        """Analyze a file or directory for quality issues."""
        if self.enhanced_mode:
            return self.enhanced_analyzer.analyze_path(path)
        
        path_obj = Path(path)
        
        if path_obj.is_file():
            return self._analyze_file(path_obj)
        elif path_obj.is_dir():
            return self._analyze_directory(path_obj)
        else:
            raise ValueError(f"Path does not exist: {path}")
    
    def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file."""
        if not LanguageDetector.is_supported_language(str(file_path)):
            return {
                'summary': {
                    'files_analyzed': 0,
                    'lines_of_code': 0,
                    'total_issues': 0
                },
                'categories': {}
            }
        
        try:
            content = file_path.read_text(encoding='utf-8')
            language = LanguageDetector.detect_language(str(file_path))
            
            # Analyze based on language
            if language == 'python':
                issues = self.python_analyzer.analyze_file(str(file_path), content)
            elif language in ['javascript', 'typescript']:
                issues = self.javascript_analyzer.analyze_file(str(file_path), content)
            else:
                issues = []
            
            # Convert issues to dictionary format
            issues_dict = [self._issue_to_dict(issue) for issue in issues]
            
            return {
                'summary': {
                    'files_analyzed': 1,
                    'lines_of_code': len(content.split('\n')),
                    'total_issues': len(issues)
                },
                'categories': self._categorize_issues(issues_dict)
            }
            
        except Exception as e:
            return {
                'summary': {
                    'files_analyzed': 0,
                    'lines_of_code': 0,
                    'total_issues': 0,
                    'error': str(e)
                },
                'categories': {}
            }
    
    def _analyze_directory(self, dir_path: Path) -> Dict[str, Any]:
        """Analyze all supported files in a directory."""
        all_issues = []
        files_analyzed = 0
        total_lines = 0
        
        # Find all supported files
        supported_files = []
        for ext in ['.py', '.js', '.ts', '.jsx', '.tsx']:
            supported_files.extend(dir_path.rglob(f'*{ext}'))
        
        # Analyze each file
        for file_path in supported_files:
            if file_path.is_file():
                try:
                    result = self._analyze_file(file_path)
                    if result['summary']['files_analyzed'] > 0:
                        files_analyzed += result['summary']['files_analyzed']
                        total_lines += result['summary']['lines_of_code']
                        
                        # Collect issues from all categories
                        for category_data in result['categories'].values():
                            all_issues.extend(category_data['issues'])
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
        
        return {
            'summary': {
                'files_analyzed': files_analyzed,
                'lines_of_code': total_lines,
                'total_issues': len(all_issues)
            },
            'categories': self._categorize_issues(all_issues)
        }
    
    def _categorize_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Categorize issues by type."""
        categories = {}
        
        for issue in issues:
            category = issue.get('category', 'general')
            if category not in categories:
                categories[category] = {
                    'name': category.title(),
                    'description': f"{category.title()} related issues",
                    'issues': []
                }
            categories[category]['issues'].append(issue)
        
        return categories
    
    def _issue_to_dict(self, issue: CodeIssue) -> Dict[str, Any]:
        """Convert CodeIssue to dictionary."""
        return {
            'message': issue.message,
            'severity': issue.severity.value,
            'file': issue.file,
            'line': issue.line,
            'column': issue.column,
            'category': issue.category,
            'suggestion': issue.suggestion
        }


class DuplicationDetector:
    """Detects code duplication across files."""
    
    def __init__(self):
        self.min_lines = 5  # Minimum lines to consider as duplication
    
    def detect_duplication(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Detect code duplication across multiple files."""
        # TODO: Implement duplication detection
        # This would involve:
        # 1. Tokenizing code blocks
        # 2. Creating fingerprints
        # 3. Finding similar blocks
        # 4. Reporting duplications
        return []
