"""
Data models for the Code Quality Intelligence Agent.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from pathlib import Path


class Severity(Enum):
    """Issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CodeIssue:
    """Represents a code quality issue."""
    message: str
    severity: Severity
    file: str
    line: Optional[int] = None
    column: Optional[int] = None
    category: str = "general"
    suggestion: Optional[str] = None


@dataclass
class PerformanceIssue:
    """Represents a performance-related issue."""
    message: str
    severity: Severity
    file: str
    line: Optional[int] = None
    category: str = "performance"
    suggestion: Optional[str] = None
    impact: str = "medium"  # low, medium, high


@dataclass
class TestingGap:
    """Represents a testing gap."""
    message: str
    severity: Severity
    file: str
    line: Optional[int] = None
    category: str = "testing"
    suggestion: Optional[str] = None
    test_type: str = "unit"  # unit, integration, e2e


class LanguageDetector:
    """Detects programming language from file extensions and content."""
    
    EXTENSION_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript'
    }
    
    @classmethod
    def detect_language(cls, file_path: str) -> Optional[str]:
        """Detect programming language from file path."""
        ext = Path(file_path).suffix.lower()
        return cls.EXTENSION_MAP.get(ext)
    
    @classmethod
    def is_supported_language(cls, file_path: str) -> bool:
        """Check if file is in a supported language."""
        language = cls.detect_language(file_path)
        return language in ['python', 'javascript', 'typescript']
