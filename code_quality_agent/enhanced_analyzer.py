"""
Enhanced analysis capabilities for the Code Quality Intelligence Agent.
"""

import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter

from .models import CodeIssue, Severity, LanguageDetector, PerformanceIssue, TestingGap




class PerformanceAnalyzer:
    """Analyzes code for performance bottlenecks."""
    
    def __init__(self):
        self.issues: List[PerformanceIssue] = []
    
    def analyze_file(self, file_path: str, content: str, language: str) -> List[PerformanceIssue]:
        """Analyze a file for performance issues."""
        self.issues = []
        
        if language == 'python':
            self._analyze_python_performance(content, file_path)
        elif language in ['javascript', 'typescript']:
            self._analyze_javascript_performance(content, file_path)
        
        return self.issues
    
    def _analyze_python_performance(self, content: str, file_path: str):
        """Analyze Python code for performance issues."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for inefficient loops
            if re.search(r'for\s+\w+\s+in\s+range\(len\(', line):
                self.issues.append(PerformanceIssue(
                    message="Inefficient loop using range(len())",
                    severity=Severity.MEDIUM,
                    file=file_path,
                    line=i,
                    suggestion="Use enumerate() or direct iteration",
                    impact="medium"
                ))
            
            # Check for string concatenation in loops
            if 'for' in line and '+' in line and ('"' in line or "'" in line):
                self.issues.append(PerformanceIssue(
                    message="String concatenation in loop detected",
                    severity=Severity.MEDIUM,
                    file=file_path,
                    line=i,
                    suggestion="Use join() or f-strings for better performance",
                    impact="high"
                ))
            
            # Check for inefficient list comprehensions
            if '[' in line and 'for' in line and 'if' in line:
                # Look for nested comprehensions
                if line.count('[') > 1:
                    self.issues.append(PerformanceIssue(
                        message="Complex nested list comprehension",
                        severity=Severity.LOW,
                        file=file_path,
                        line=i,
                        suggestion="Consider breaking into multiple steps for readability",
                        impact="low"
                    ))
            
            # Check for global variable access in functions
            if 'def ' in line and 'global ' in content:
                self.issues.append(PerformanceIssue(
                    message="Global variable access in function",
                    severity=Severity.MEDIUM,
                    file=file_path,
                    line=i,
                    suggestion="Pass variables as parameters instead of using global",
                    impact="medium"
                ))
    
    def _analyze_javascript_performance(self, content: str, file_path: str):
        """Analyze JavaScript/TypeScript code for performance issues."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for inefficient DOM queries
            if re.search(r'getElementById\(.*\)\.getElementById\(', line):
                self.issues.append(PerformanceIssue(
                    message="Chained DOM queries detected",
                    severity=Severity.MEDIUM,
                    file=file_path,
                    line=i,
                    suggestion="Cache DOM elements or use querySelector",
                    impact="medium"
                ))
            
            # Check for inefficient array operations
            if 'for' in line and 'length' in line and 'i++' in line:
                self.issues.append(PerformanceIssue(
                    message="Traditional for loop with length check",
                    severity=Severity.LOW,
                    file=file_path,
                    line=i,
                    suggestion="Consider using for...of or array methods",
                    impact="low"
                ))
            
            # Check for synchronous operations
            if any(op in line for op in ['XMLHttpRequest', 'fs.readFileSync', 'require(']):
                self.issues.append(PerformanceIssue(
                    message="Synchronous operation detected",
                    severity=Severity.HIGH,
                    file=file_path,
                    line=i,
                    suggestion="Use async/await or promises for better performance",
                    impact="high"
                ))
            
            # Check for memory leaks
            if 'addEventListener' in line and 'removeEventListener' not in content:
                self.issues.append(PerformanceIssue(
                    message="Potential memory leak: event listener without cleanup",
                    severity=Severity.MEDIUM,
                    file=file_path,
                    line=i,
                    suggestion="Ensure event listeners are properly removed",
                    impact="medium"
                ))


class TestingAnalyzer:
    """Analyzes code for testing gaps and coverage."""
    
    def __init__(self):
        self.gaps: List[TestingGap] = []
    
    def analyze_file(self, file_path: str, content: str, language: str) -> List[TestingGap]:
        """Analyze a file for testing gaps."""
        self.gaps = []
        
        if language == 'python':
            self._analyze_python_testing(content, file_path)
        elif language in ['javascript', 'typescript']:
            self._analyze_javascript_testing(content, file_path)
        
        return self.gaps
    
    def _analyze_python_testing(self, content: str, file_path: str):
        """Analyze Python code for testing gaps."""
        lines = content.split('\n')
        
        # Find functions and classes
        functions = []
        classes = []
        
        for i, line in enumerate(lines, 1):
            if re.match(r'^\s*def\s+\w+', line):
                func_name = re.search(r'def\s+(\w+)', line).group(1)
                if not func_name.startswith('_'):
                    functions.append((i, func_name))
            elif re.match(r'^\s*class\s+\w+', line):
                class_name = re.search(r'class\s+(\w+)', line).group(1)
                classes.append((i, class_name))
        
        # Check for test files
        is_test_file = 'test' in file_path.lower() or file_path.endswith('_test.py')
        
        if not is_test_file and (functions or classes):
            # Check if there are corresponding test files
            test_files = self._find_test_files(file_path)
            
            if not test_files:
                self.gaps.append(TestingGap(
                    message=f"No test files found for {file_path}",
                    severity=Severity.HIGH,
                    file=file_path,
                    suggestion="Create test files to ensure code quality",
                    test_type="unit"
                ))
            
            # Check for complex functions without tests
            for line_num, func_name in functions:
                if self._is_complex_function(lines, line_num):
                    self.gaps.append(TestingGap(
                        message=f"Complex function '{func_name}' may need comprehensive testing",
                        severity=Severity.MEDIUM,
                        file=file_path,
                        line=line_num,
                        suggestion="Add unit tests for edge cases and error conditions",
                        test_type="unit"
                    ))
    
    def _analyze_javascript_testing(self, content: str, file_path: str):
        """Analyze JavaScript/TypeScript code for testing gaps."""
        lines = content.split('\n')
        
        # Find functions and classes
        functions = []
        classes = []
        
        for i, line in enumerate(lines, 1):
            if re.search(r'function\s+\w+', line) or re.search(r'const\s+\w+\s*=\s*\(', line):
                functions.append((i, "function"))
            elif re.search(r'class\s+\w+', line):
                class_name = re.search(r'class\s+(\w+)', line).group(1)
                classes.append((i, class_name))
        
        # Check for test files
        is_test_file = any(x in file_path.lower() for x in ['test', 'spec', '.test.', '.spec.'])
        
        if not is_test_file and (functions or classes):
            test_files = self._find_test_files(file_path)
            
            if not test_files:
                self.gaps.append(TestingGap(
                    message=f"No test files found for {file_path}",
                    severity=Severity.HIGH,
                    file=file_path,
                    suggestion="Create test files using Jest, Mocha, or similar framework",
                    test_type="unit"
                ))
    
    def _find_test_files(self, file_path: str) -> List[str]:
        """Find corresponding test files."""
        file_path_obj = Path(file_path)
        parent_dir = file_path_obj.parent
        base_name = file_path_obj.stem
        
        test_files = []
        
        # Look for common test file patterns
        test_patterns = [
            f"test_{base_name}.py",
            f"{base_name}_test.py",
            f"test_{base_name}.js",
            f"{base_name}.test.js",
            f"{base_name}.spec.js"
        ]
        
        for pattern in test_patterns:
            test_file = parent_dir / pattern
            if test_file.exists():
                test_files.append(str(test_file))
        
        return test_files
    
    def _is_complex_function(self, lines: List[str], start_line: int) -> bool:
        """Check if a function is complex enough to warrant special testing."""
        if start_line >= len(lines):
            return False
        
        # Simple heuristic: count lines, branches, and complexity indicators
        function_lines = 0
        branches = 0
        
        for i in range(start_line, min(start_line + 50, len(lines))):
            line = lines[i].strip()
            
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('def ') and i > start_line:
                break
            
            function_lines += 1
            
            # Count branches
            if any(keyword in line for keyword in ['if', 'elif', 'else', 'for', 'while', 'try', 'except']):
                branches += 1
        
        # Consider complex if more than 20 lines or more than 5 branches
        return function_lines > 20 or branches > 5


class DocumentationAnalyzer:
    """Analyzes code for documentation coverage and quality."""
    
    def __init__(self):
        self.issues: List[CodeIssue] = []
    
    def analyze_file(self, file_path: str, content: str, language: str) -> List[CodeIssue]:
        """Analyze a file for documentation issues."""
        self.issues = []
        
        if language == 'python':
            self._analyze_python_documentation(content, file_path)
        elif language in ['javascript', 'typescript']:
            self._analyze_javascript_documentation(content, file_path)
        
        return self.issues
    
    def _analyze_python_documentation(self, content: str, file_path: str):
        """Analyze Python code for documentation issues."""
        lines = content.split('\n')
        
        # Check for module docstring
        if not self._has_module_docstring(lines):
            self.issues.append(CodeIssue(
                message="Module missing docstring",
                severity=Severity.LOW,
                file=file_path,
                line=1,
                category="documentation",
                suggestion="Add a module docstring describing the file's purpose"
            ))
        
        # Check for class and function docstrings
        for i, line in enumerate(lines, 1):
            if re.match(r'^\s*def\s+\w+', line):
                func_name = re.search(r'def\s+(\w+)', line).group(1)
                if not func_name.startswith('_') and not self._has_docstring(lines, i):
                    self.issues.append(CodeIssue(
                        message=f"Function '{func_name}' missing docstring",
                        severity=Severity.LOW,
                        file=file_path,
                        line=i,
                        category="documentation",
                        suggestion="Add a docstring describing parameters, return value, and behavior"
                    ))
            
            elif re.match(r'^\s*class\s+\w+', line):
                class_name = re.search(r'class\s+(\w+)', line).group(1)
                if not class_name.startswith('_') and not self._has_docstring(lines, i):
                    self.issues.append(CodeIssue(
                        message=f"Class '{class_name}' missing docstring",
                        severity=Severity.LOW,
                        file=file_path,
                        line=i,
                        category="documentation",
                        suggestion="Add a class docstring describing purpose and usage"
                    ))
    
    def _analyze_javascript_documentation(self, content: str, file_path: str):
        """Analyze JavaScript/TypeScript code for documentation issues."""
        lines = content.split('\n')
        
        # Check for JSDoc comments
        for i, line in enumerate(lines, 1):
            if re.search(r'function\s+\w+', line) or re.search(r'const\s+\w+\s*=\s*\(', line):
                # Check if there's a JSDoc comment above
                has_jsdoc = False
                for j in range(max(0, i-5), i):
                    if '/**' in lines[j] or '* @' in lines[j]:
                        has_jsdoc = True
                        break
                
                if not has_jsdoc:
                    self.issues.append(CodeIssue(
                        message="Function missing JSDoc documentation",
                        severity=Severity.LOW,
                        file=file_path,
                        line=i,
                        category="documentation",
                        suggestion="Add JSDoc comments describing parameters and return value"
                    ))
    
    def _has_module_docstring(self, lines: List[str]) -> bool:
        """Check if the module has a docstring."""
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith('"""') or line.startswith("'''"):
                return True
            if line and not line.startswith('#'):
                break
        return False
    
    def _has_docstring(self, lines: List[str], start_line: int) -> bool:
        """Check if a function/class has a docstring."""
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


class DuplicationDetector:
    """Enhanced duplication detection with similarity analysis."""
    
    def __init__(self):
        self.min_lines = 5
        self.similarity_threshold = 0.8
    
    def detect_duplication(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Detect code duplication across files."""
        duplications = []
        
        # Group files by language
        python_files = [f for f in files if LanguageDetector.detect_language(str(f)) == 'python']
        js_files = [f for f in files if LanguageDetector.detect_language(str(f)) in ['javascript', 'typescript']]
        
        # Check within each language group
        if python_files:
            duplications.extend(self._check_duplication_in_group(python_files))
        if js_files:
            duplications.extend(self._check_duplication_in_group(js_files))
        
        return duplications
    
    def _check_duplication_in_group(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Check for duplication within a group of files."""
        duplications = []
        
        # Read all files
        file_contents = {}
        for file_path in files:
            try:
                content = file_path.read_text(encoding='utf-8')
                file_contents[file_path] = content.split('\n')
            except Exception:
                continue
        
        # Compare files pairwise
        file_list = list(file_contents.keys())
        for i in range(len(file_list)):
            for j in range(i + 1, len(file_list)):
                file1, file2 = file_list[i], file_list[j]
                content1, content2 = file_contents[file1], file_contents[file2]
                
                # Find similar blocks
                similar_blocks = self._find_similar_blocks(content1, content2)
                
                for block in similar_blocks:
                    if block['similarity'] >= self.similarity_threshold:
                        duplications.append({
                            'message': f"Code duplication detected between {file1.name} and {file2.name}",
                            'severity': 'medium',
                            'file1': str(file1),
                            'file2': str(file2),
                            'lines1': f"{block['start1']}-{block['end1']}",
                            'lines2': f"{block['start2']}-{block['end2']}",
                            'similarity': block['similarity'],
                            'category': 'duplication',
                            'suggestion': 'Consider extracting common code into a shared function or module'
                        })
        
        return duplications
    
    def _find_similar_blocks(self, content1: List[str], content2: List[str]) -> List[Dict[str, Any]]:
        """Find similar blocks between two file contents."""
        similar_blocks = []
        
        # Simple sliding window approach
        window_size = self.min_lines
        
        for i in range(len(content1) - window_size + 1):
            block1 = content1[i:i + window_size]
            
            for j in range(len(content2) - window_size + 1):
                block2 = content2[j:j + window_size]
                
                similarity = self._calculate_similarity(block1, block2)
                
                if similarity >= self.similarity_threshold:
                    similar_blocks.append({
                        'start1': i + 1,
                        'end1': i + window_size,
                        'start2': j + 1,
                        'end2': j + window_size,
                        'similarity': similarity
                    })
        
        return similar_blocks
    
    def _calculate_similarity(self, block1: List[str], block2: List[str]) -> float:
        """Calculate similarity between two code blocks."""
        if len(block1) != len(block2):
            return 0.0
        
        # Normalize lines (remove whitespace, comments)
        norm1 = [self._normalize_line(line) for line in block1]
        norm2 = [self._normalize_line(line) for line in block2]
        
        # Calculate Jaccard similarity
        set1 = set(norm1)
        set2 = set(norm2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _normalize_line(self, line: str) -> str:
        """Normalize a line for comparison."""
        # Remove comments
        line = re.sub(r'#.*$', '', line)
        line = re.sub(r'//.*$', '', line)
        
        # Remove extra whitespace
        line = re.sub(r'\s+', ' ', line.strip())
        
        # Remove variable names (replace with placeholders)
        line = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', 'VAR', line)
        
        return line


class EnhancedCodeAnalyzer:
    """Enhanced analyzer that combines all analysis capabilities."""
    
    def __init__(self):
        # Import original analyzers to avoid circular imports
        from .analyzer import PythonAnalyzer, JavaScriptAnalyzer
        self.python_analyzer = PythonAnalyzer()
        self.javascript_analyzer = JavaScriptAnalyzer()
        
        # Enhanced analyzers
        self.performance_analyzer = PerformanceAnalyzer()
        self.testing_analyzer = TestingAnalyzer()
        self.documentation_analyzer = DocumentationAnalyzer()
        self.duplication_detector = DuplicationDetector()
    
    def analyze_path(self, path: str) -> Dict[str, Any]:
        """Enhanced analysis of a file or directory."""
        path_obj = Path(path)
        
        if path_obj.is_file():
            return self._analyze_file_enhanced(path_obj)
        elif path_obj.is_dir():
            return self._analyze_directory_enhanced(path_obj)
        else:
            raise ValueError(f"Path does not exist: {path}")
    
    def _analyze_file_enhanced(self, file_path: Path) -> Dict[str, Any]:
        """Enhanced analysis of a single file."""
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
            
            # Collect all issues
            all_issues = []
            
            # Original analyzers (security, complexity, style)
            if language == 'python':
                original_issues = self.python_analyzer.analyze_file(str(file_path), content)
            elif language in ['javascript', 'typescript']:
                original_issues = self.javascript_analyzer.analyze_file(str(file_path), content)
            else:
                original_issues = []
            
            all_issues.extend([self._issue_to_dict(issue) for issue in original_issues])
            
            # Performance analysis
            performance_issues = self.performance_analyzer.analyze_file(str(file_path), content, language)
            all_issues.extend([self._performance_to_dict(issue) for issue in performance_issues])
            
            # Testing analysis
            testing_gaps = self.testing_analyzer.analyze_file(str(file_path), content, language)
            all_issues.extend([self._testing_to_dict(gap) for gap in testing_gaps])
            
            # Documentation analysis
            doc_issues = self.documentation_analyzer.analyze_file(str(file_path), content, language)
            all_issues.extend([self._issue_to_dict(issue) for issue in doc_issues])
            
            return {
                'summary': {
                    'files_analyzed': 1,
                    'lines_of_code': len(content.split('\n')),
                    'total_issues': len(all_issues)
                },
                'categories': self._categorize_issues(all_issues)
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
    
    def _analyze_directory_enhanced(self, dir_path: Path) -> Dict[str, Any]:
        """Enhanced analysis of a directory."""
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
                    result = self._analyze_file_enhanced(file_path)
                    if result['summary']['files_analyzed'] > 0:
                        files_analyzed += result['summary']['files_analyzed']
                        total_lines += result['summary']['lines_of_code']
                        
                        # Collect issues from all categories
                        for category_data in result['categories'].values():
                            all_issues.extend(category_data['issues'])
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
        
        # Check for duplication across files
        if len(supported_files) > 1:
            duplications = self.duplication_detector.detect_duplication(supported_files)
            all_issues.extend(duplications)
        
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
    
    def _performance_to_dict(self, issue: PerformanceIssue) -> Dict[str, Any]:
        """Convert PerformanceIssue to dictionary."""
        return {
            'message': issue.message,
            'severity': issue.severity.value,
            'file': issue.file,
            'line': issue.line,
            'category': issue.category,
            'suggestion': issue.suggestion,
            'impact': issue.impact
        }
    
    def _testing_to_dict(self, gap: TestingGap) -> Dict[str, Any]:
        """Convert TestingGap to dictionary."""
        return {
            'message': gap.message,
            'severity': gap.severity.value,
            'file': gap.file,
            'line': gap.line,
            'category': gap.category,
            'suggestion': gap.suggestion,
            'test_type': gap.test_type
        }
    
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
