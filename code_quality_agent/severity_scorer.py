"""
Automated severity scoring and prioritization system.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re


class ImpactLevel(Enum):
    """Impact levels for issues."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class LikelihoodLevel(Enum):
    """Likelihood levels for issues."""
    UNLIKELY = 1
    POSSIBLE = 2
    LIKELY = 3
    CERTAIN = 4


@dataclass
class SeverityScore:
    """Represents a calculated severity score."""
    impact: ImpactLevel
    likelihood: LikelihoodLevel
    score: float
    priority: str
    reasoning: str


class SeverityScorer:
    """Automated severity scoring system."""
    
    def __init__(self):
        # Define patterns for different severity levels
        self.critical_patterns = [
            r'eval\(',
            r'exec\(',
            r'__import__\(',
            r'shell=True',
            r'subprocess\.call.*shell=True',
            r'os\.system\(',
            r'pickle\.loads\(',
            r'yaml\.load\(',
            r'SQL.*\+.*%s',
            r'innerHTML.*=.*\+',
            r'document\.write\(',
            r'eval\(',
            r'Function\(',
            r'setTimeout.*eval',
            r'setInterval.*eval'
        ]
        
        self.high_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'\.innerHTML\s*=',
            r'\.outerHTML\s*=',
            r'document\.cookie',
            r'localStorage\.setItem',
            r'sessionStorage\.setItem',
            r'XMLHttpRequest',
            r'fetch\(',
            r'axios\(',
            r'requests\.get\(',
            r'requests\.post\(',
            r'urllib\.request',
            r'http\.client',
            r'for\s+\w+\s+in\s+range\(len\(',
            r'while\s+True:',
            r'while\s+1:',
            r'for\s+\(;;\)',
            r'recursive.*function',
            r'deep.*copy',
            r'pickle\.dump',
            r'json\.loads',
            r'yaml\.load'
        ]
        
        self.medium_patterns = [
            r'console\.log\(',
            r'print\(',
            r'debugger',
            r'var\s+\w+',
            r'==\s*[^=]',
            r'!=\s*[^=]',
            r'function\s+\w+.*{.*{.*{.*{',
            r'if.*{.*if.*{.*if.*{',
            r'for.*{.*for.*{.*for.*{',
            r'while.*{.*while.*{',
            r'try.*{.*try.*{',
            r'catch.*{.*catch.*{',
            r'def\s+\w+\([^)]{50,}\)',
            r'function\s+\w+\([^)]{50,}\)',
            r'class\s+\w+.*{.*{.*{.*{',
            r'import\s+\*',
            r'from\s+\w+\s+import\s+\*',
            r'require\(.*\*',
            r'global\s+\w+',
            r'nonlocal\s+\w+'
        ]
        
        self.low_patterns = [
            r'# TODO',
            r'# FIXME',
            r'# HACK',
            r'# XXX',
            r'// TODO',
            r'// FIXME',
            r'// HACK',
            r'// XXX',
            r'/\* TODO',
            r'/\* FIXME',
            r'/\* HACK',
            r'/\* XXX',
            r'def\s+\w+\(\):',
            r'function\s+\w+\(\)\s*{',
            r'const\s+\w+\s*=\s*\(\)\s*=>',
            r'let\s+\w+\s*=\s*\(\)\s*=>',
            r'var\s+\w+\s*=\s*\(\)\s*=>',
            r'class\s+\w+\s*{',
            r'interface\s+\w+\s*{',
            r'type\s+\w+\s*=',
            r'enum\s+\w+\s*{'
        ]
    
    def calculate_severity(self, issue: Dict[str, Any]) -> SeverityScore:
        """Calculate severity score for an issue."""
        message = issue.get('message', '')
        category = issue.get('category', 'general')
        file_path = issue.get('file', '')
        line = issue.get('line', 0)
        
        # Calculate impact
        impact = self._calculate_impact(message, category, file_path)
        
        # Calculate likelihood
        likelihood = self._calculate_likelihood(message, category, file_path, line)
        
        # Calculate final score
        score = self._calculate_score(impact, likelihood)
        
        # Determine priority
        priority = self._determine_priority(score)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(impact, likelihood, score, priority)
        
        return SeverityScore(
            impact=impact,
            likelihood=likelihood,
            score=score,
            priority=priority,
            reasoning=reasoning
        )
    
    def _calculate_impact(self, message: str, category: str, file_path: str) -> ImpactLevel:
        """Calculate impact level based on issue characteristics."""
        message_lower = message.lower()
        category_lower = category.lower()
        
        # Check for critical patterns
        for pattern in self.critical_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return ImpactLevel.CRITICAL
        
        # Category-based impact assessment
        if category_lower in ['security', 'vulnerability']:
            if any(keyword in message_lower for keyword in ['injection', 'xss', 'csrf', 'rce', 'lfi', 'rfi']):
                return ImpactLevel.CRITICAL
            elif any(keyword in message_lower for keyword in ['password', 'secret', 'key', 'token']):
                return ImpactLevel.HIGH
            else:
                return ImpactLevel.MEDIUM
        
        elif category_lower in ['performance']:
            if any(keyword in message_lower for keyword in ['memory leak', 'infinite loop', 'deadlock']):
                return ImpactLevel.HIGH
            elif any(keyword in message_lower for keyword in ['slow', 'inefficient', 'bottleneck']):
                return ImpactLevel.MEDIUM
            else:
                return ImpactLevel.LOW
        
        elif category_lower in ['testing']:
            if 'no test' in message_lower or 'missing test' in message_lower:
                return ImpactLevel.HIGH
            elif 'test coverage' in message_lower:
                return ImpactLevel.MEDIUM
            else:
                return ImpactLevel.LOW
        
        elif category_lower in ['documentation']:
            return ImpactLevel.LOW
        
        elif category_lower in ['complexity']:
            if any(keyword in message_lower for keyword in ['too complex', 'too long', 'too many']):
                return ImpactLevel.MEDIUM
            else:
                return ImpactLevel.LOW
        
        elif category_lower in ['style', 'formatting']:
            return ImpactLevel.LOW
        
        # Check for high-impact patterns
        for pattern in self.high_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return ImpactLevel.HIGH
        
        # Check for medium-impact patterns
        for pattern in self.medium_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return ImpactLevel.MEDIUM
        
        # Default to low impact
        return ImpactLevel.LOW
    
    def _calculate_likelihood(self, message: str, category: str, file_path: str, line: int) -> LikelihoodLevel:
        """Calculate likelihood of the issue occurring."""
        message_lower = message.lower()
        category_lower = category.lower()
        
        # File-based likelihood assessment
        if 'test' in file_path.lower():
            return LikelihoodLevel.UNLIKELY  # Issues in test files are less likely to affect production
        
        if any(keyword in file_path.lower() for keyword in ['config', 'settings', 'env']):
            return LikelihoodLevel.CERTAIN  # Config issues are certain to affect the system
        
        if any(keyword in file_path.lower() for keyword in ['main', 'app', 'index', 'server']):
            return LikelihoodLevel.LIKELY  # Main files are likely to be executed
        
        # Category-based likelihood
        if category_lower in ['security']:
            return LikelihoodLevel.LIKELY  # Security issues are likely to be exploited
        
        elif category_lower in ['performance']:
            return LikelihoodLevel.POSSIBLE  # Performance issues may or may not manifest
        
        elif category_lower in ['testing']:
            return LikelihoodLevel.CERTAIN  # Testing gaps are certain to exist if detected
        
        elif category_lower in ['documentation']:
            return LikelihoodLevel.POSSIBLE  # Documentation issues may or may not affect users
        
        elif category_lower in ['complexity']:
            return LikelihoodLevel.LIKELY  # Complexity issues are likely to cause problems
        
        elif category_lower in ['style', 'formatting']:
            return LikelihoodLevel.UNLIKELY  # Style issues rarely cause functional problems
        
        # Message-based likelihood
        if any(keyword in message_lower for keyword in ['always', 'never', 'every', 'all']):
            return LikelihoodLevel.CERTAIN
        
        if any(keyword in message_lower for keyword in ['may', 'might', 'could', 'possibly']):
            return LikelihoodLevel.POSSIBLE
        
        if any(keyword in message_lower for keyword in ['will', 'should', 'must', 'required']):
            return LikelihoodLevel.LIKELY
        
        # Default likelihood
        return LikelihoodLevel.POSSIBLE
    
    def _calculate_score(self, impact: ImpactLevel, likelihood: LikelihoodLevel) -> float:
        """Calculate final severity score."""
        # Weighted combination of impact and likelihood
        impact_weight = 0.7
        likelihood_weight = 0.3
        
        score = (impact.value * impact_weight + likelihood.value * likelihood_weight) / 4.0
        
        return round(score, 2)
    
    def _determine_priority(self, score: float) -> str:
        """Determine priority based on score."""
        if score >= 0.8:
            return "P0 - Critical"
        elif score >= 0.6:
            return "P1 - High"
        elif score >= 0.4:
            return "P2 - Medium"
        elif score >= 0.2:
            return "P3 - Low"
        else:
            return "P4 - Info"
    
    def _generate_reasoning(self, impact: ImpactLevel, likelihood: LikelihoodLevel, score: float, priority: str) -> str:
        """Generate human-readable reasoning for the severity score."""
        impact_desc = {
            ImpactLevel.CRITICAL: "critical impact",
            ImpactLevel.HIGH: "high impact",
            ImpactLevel.MEDIUM: "medium impact",
            ImpactLevel.LOW: "low impact"
        }
        
        likelihood_desc = {
            LikelihoodLevel.CERTAIN: "certain to occur",
            LikelihoodLevel.LIKELY: "likely to occur",
            LikelihoodLevel.POSSIBLE: "possibly occurring",
            LikelihoodLevel.UNLIKELY: "unlikely to occur"
        }
        
        reasoning = f"Issue has {impact_desc[impact]} and is {likelihood_desc[likelihood]}. "
        reasoning += f"Severity score: {score}/1.0. Priority: {priority}."
        
        return reasoning
    
    def prioritize_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize a list of issues based on severity scores."""
        scored_issues = []
        
        for issue in issues:
            severity_score = self.calculate_severity(issue)
            
            # Add severity information to the issue
            enhanced_issue = issue.copy()
            enhanced_issue.update({
                'severity_score': severity_score.score,
                'impact_level': severity_score.impact.value,
                'likelihood_level': severity_score.likelihood.value,
                'priority': severity_score.priority,
                'reasoning': severity_score.reasoning
            })
            
            scored_issues.append(enhanced_issue)
        
        # Sort by severity score (descending)
        scored_issues.sort(key=lambda x: x['severity_score'], reverse=True)
        
        return scored_issues
    
    def get_priority_summary(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get a summary of issues by priority level."""
        priority_counts = {
            "P0 - Critical": 0,
            "P1 - High": 0,
            "P2 - Medium": 0,
            "P3 - Low": 0,
            "P4 - Info": 0
        }
        
        for issue in issues:
            priority = issue.get('priority', 'P4 - Info')
            if priority in priority_counts:
                priority_counts[priority] += 1
        
        return priority_counts
    
    def get_risk_assessment(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall risk assessment."""
        if not issues:
            return {
                'risk_level': 'Low',
                'risk_score': 0.0,
                'recommendations': ['No issues found. Code quality is good.']
            }
        
        # Calculate average severity score
        avg_score = sum(issue.get('severity_score', 0) for issue in issues) / len(issues)
        
        # Count critical and high priority issues
        critical_count = sum(1 for issue in issues if issue.get('priority', '').startswith('P0'))
        high_count = sum(1 for issue in issues if issue.get('priority', '').startswith('P1'))
        
        # Determine risk level
        if critical_count > 0:
            risk_level = 'Critical'
        elif high_count > 3:
            risk_level = 'High'
        elif avg_score > 0.6:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Generate recommendations
        recommendations = []
        
        if critical_count > 0:
            recommendations.append(f"🚨 {critical_count} critical issues require immediate attention")
        
        if high_count > 0:
            recommendations.append(f"⚠️ {high_count} high-priority issues should be addressed soon")
        
        if avg_score > 0.5:
            recommendations.append("📊 Consider implementing automated code quality checks")
        
        if len(issues) > 20:
            recommendations.append("🔍 Large number of issues detected - consider code review process")
        
        recommendations.append("💡 Focus on security and performance issues first")
        
        return {
            'risk_level': risk_level,
            'risk_score': round(avg_score, 2),
            'total_issues': len(issues),
            'critical_issues': critical_count,
            'high_issues': high_count,
            'recommendations': recommendations
        }

