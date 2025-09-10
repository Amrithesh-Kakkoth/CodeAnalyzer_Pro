"""
Enhanced Interactive CLI for Code Quality Intelligence Agent.
Provides conversational Q&A experience with better UX.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.align import Align

from .qa_agent import CodeQAAgent
from .analyzer import CodeAnalyzer
from .severity_scorer import SeverityScorer
from .config import config


class InteractiveCLI:
    """Enhanced interactive CLI for code quality Q&A."""
    
    def __init__(self, codebase_path: str):
        self.codebase_path = codebase_path
        self.is_github_repo = codebase_path.startswith(('http://github.com/', 'https://github.com/'))
        self.console = Console()
        self.qa_agent: Optional[CodeQAAgent] = None
        self.analyzer: Optional[CodeAnalyzer] = None
        self.scorer: Optional[SeverityScorer] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.analysis_results: Optional[Dict[str, Any]] = None
        
    def initialize(self) -> bool:
        """Initialize the AI agent and analysis components."""
        try:
            # Check API key
            if not self._check_api_key():
                return False
            
            # Initialize components
            with self.console.status("[bold green]Initializing AI agent..."):
                self.qa_agent = CodeQAAgent(str(self.codebase_path))
                
            with self.console.status("[bold green]Initializing analyzer..."):
                self.analyzer = CodeAnalyzer(enhanced_mode=True)
                self.scorer = SeverityScorer()
                
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error initializing: {e}[/red]")
            return False
    
    def _check_api_key(self) -> bool:
        """Check if API key is configured."""
        if config.ai.llm_provider == "groq" and not config.ai.groq_api_key:
            self.console.print("[red]Error: Groq API key not found.[/red]")
            self.console.print("[yellow]Please set GROQ_API_KEY environment variable.[/yellow]")
            return False
        elif config.ai.llm_provider == "openai" and not config.ai.openai_api_key:
            self.console.print("[red]Error: OpenAI API key not found.[/red]")
            self.console.print("[yellow]Please set OPENAI_API_KEY environment variable.[/yellow]")
            return False
        return True
    
    def show_welcome(self):
        """Display welcome screen with codebase info."""
        # Get codebase info
        if self.is_github_repo:
            codebase_info = f"🌐 GitHub Repository: {self.codebase_path}"
            file_count = "Unknown"  # Will be updated after analysis
        elif Path(self.codebase_path).is_file():
            codebase_info = f"📄 File: {Path(self.codebase_path).name}"
            file_count = 1
        else:
            codebase_info = f"📁 Directory: {Path(self.codebase_path).name}"
            # Count files
            supported_files = []
            for ext in ['.py', '.js', '.ts', '.jsx', '.tsx']:
                supported_files.extend(Path(self.codebase_path).rglob(f'*{ext}'))
            file_count = len(supported_files)
        
        welcome_text = f"""
🤖 **Code Quality Intelligence Agent**

{codebase_info}
📊 **Files to analyze:** {file_count}
🧠 **AI Model:** {config.ai.groq_model_name if config.ai.llm_provider == 'groq' else config.ai.openai_model_name}
🔍 **Provider:** {config.ai.llm_provider.upper()}

**Ask me anything about your code!**
• Security vulnerabilities
• Performance issues  
• Code architecture
• Best practices
• Refactoring suggestions
• Testing strategies

Type `help` for more commands, `quit` to exit.
        """
        
        self.console.print(Panel(
            Markdown(welcome_text),
            title="🚀 Welcome",
            border_style="blue"
        ))
    
    def show_help(self):
        """Display help information."""
        help_text = """
## 🆘 **Available Commands**

### **Analysis Commands**
- `analyze` - Run full code quality analysis
- `security` - Focus on security vulnerabilities
- `performance` - Analyze performance issues
- `complexity` - Check code complexity
- `documentation` - Review documentation gaps
- `testing` - Identify testing needs

### **Interactive Commands**
- `summary` - Get codebase overview
- `priorities` - Show prioritized issues
- `trends` - Analyze quality trends
- `recommendations` - Get improvement suggestions

### **Utility Commands**
- `help` - Show this help
- `clear` - Clear conversation history
- `history` - Show conversation history
- `export` - Export analysis results
- `quit` / `exit` - Exit the application

### **Example Questions**
- "What are the main security risks?"
- "How can I improve performance?"
- "What's the most complex function?"
- "What testing should I add?"
- "How would you refactor this code?"
        """
        
        self.console.print(Panel(
            Markdown(help_text),
            title="📚 Help",
            border_style="green"
        ))
    
    def run_analysis(self) -> bool:
        """Run full code quality analysis."""
        try:
            with self.console.status("[bold green]Analyzing codebase..."):
                self.analysis_results = self.analyzer.analyze_path(str(self.codebase_path))
            
            # Display results
            self._display_analysis_summary()
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error during analysis: {e}[/red]")
            return False
    
    def _display_analysis_summary(self):
        """Display analysis summary."""
        if not self.analysis_results:
            return
        
        summary = self.analysis_results['summary']
        categories = self.analysis_results['categories']
        
        # Create summary table
        table = Table(title="📊 Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Files Analyzed", str(summary['files_analyzed']))
        table.add_row("Lines of Code", f"{summary['lines_of_code']:,}")
        table.add_row("Total Issues", str(summary['total_issues']))
        
        # Add category breakdown
        for category, data in categories.items():
            count = len(data['issues'])
            table.add_row(f"{category.title()} Issues", str(count))
        
        self.console.print(table)
        
        # Show top issues
        if summary['total_issues'] > 0:
            self.console.print("\n[bold]🔍 Top Issues:[/bold]")
            all_issues = []
            for category_data in categories.values():
                all_issues.extend(category_data['issues'])
            
            # Prioritize issues
            prioritized = self.scorer.prioritize_issues(all_issues)
            
            for i, issue in enumerate(prioritized[:5], 1):
                severity = issue.get('severity', 'unknown').upper()
                message = issue.get('message', 'No message')
                file_path = issue.get('file', 'Unknown')
                line = issue.get('line', 'N/A')
                
                self.console.print(f"{i}. [{severity}] {message}")
                self.console.print(f"   📁 {Path(file_path).name}:{line}")
    
    def ask_question(self, question: str) -> str:
        """Ask a question to the AI agent."""
        if not self.qa_agent:
            return "Error: AI agent not initialized"
        
        try:
            with self.console.status("[bold green]AI is thinking..."):
                answer = self.qa_agent.ask_question(question)
            
            # Store in conversation history
            self.conversation_history.append({
                "question": question,
                "answer": answer,
                "timestamp": time.time()
            })
            
            return answer
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def handle_command(self, command: str) -> bool:
        """Handle special commands."""
        command = command.lower().strip()
        
        if command in ['quit', 'exit', 'q']:
            return False
        
        elif command == 'help':
            self.show_help()
            
        elif command == 'analyze':
            self.run_analysis()
            
        elif command == 'security':
            answer = self.ask_question("What security vulnerabilities do you see in this codebase?")
            self._display_answer("Security Analysis", answer)
            
        elif command == 'performance':
            answer = self.ask_question("What performance issues can you identify in this code?")
            self._display_answer("Performance Analysis", answer)
            
        elif command == 'complexity':
            answer = self.ask_question("What are the most complex parts of this codebase?")
            self._display_answer("Complexity Analysis", answer)
            
        elif command == 'documentation':
            answer = self.ask_question("What documentation issues do you see?")
            self._display_answer("Documentation Analysis", answer)
            
        elif command == 'testing':
            answer = self.ask_question("What testing gaps do you identify?")
            self._display_answer("Testing Analysis", answer)
            
        elif command == 'summary':
            answer = self.ask_question("Provide a comprehensive summary of this codebase including its main purpose, key components, and overall architecture.")
            self._display_answer("Codebase Summary", answer)
            
        elif command == 'priorities':
            if self.analysis_results:
                self._show_priorities()
            else:
                self.console.print("[yellow]Run 'analyze' first to see prioritized issues.[/yellow]")
                
        elif command == 'recommendations':
            answer = self.ask_question("What are your top recommendations for improving this codebase?")
            self._display_answer("Recommendations", answer)
            
        elif command == 'clear':
            self.conversation_history.clear()
            self.console.print("[green]Conversation history cleared.[/green]")
            
        elif command == 'history':
            self._show_history()
            
        elif command == 'export':
            self._export_results()
            
        else:
            self.console.print(f"[yellow]Unknown command: {command}[/yellow]")
            self.console.print("[blue]Type 'help' for available commands.[/blue]")
        
        return True
    
    def _display_answer(self, title: str, answer: str):
        """Display AI answer in a formatted panel."""
        # Format answer as markdown
        formatted_answer = Markdown(answer)
        
        self.console.print(Panel(
            formatted_answer,
            title=f"🤖 {title}",
            border_style="green"
        ))
    
    def _show_priorities(self):
        """Show prioritized issues."""
        if not self.analysis_results:
            return
        
        all_issues = []
        for category_data in self.analysis_results['categories'].values():
            all_issues.extend(category_data['issues'])
        
        prioritized = self.scorer.prioritize_issues(all_issues)
        risk_assessment = self.scorer.get_risk_assessment(prioritized)
        
        # Risk summary
        risk_text = f"""
**Risk Level:** {risk_assessment['risk_level']}
**Risk Score:** {risk_assessment['risk_score']}/1.0
**Critical Issues:** {risk_assessment['critical_issues']}
**High Issues:** {risk_assessment['high_issues']}
        """
        
        self.console.print(Panel(
            Markdown(risk_text),
            title="🚨 Risk Assessment",
            border_style="red"
        ))
        
        # Top priorities
        self.console.print("\n[bold]🏆 Top Priority Issues:[/bold]")
        for i, issue in enumerate(prioritized[:10], 1):
            priority = issue.get('priority', 'Unknown')
            message = issue.get('message', 'No message')
            score = issue.get('severity_score', 0)
            
            self.console.print(f"{i}. [{priority}] {message}")
            self.console.print(f"   Score: {score:.2f}")
    
    def _show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            self.console.print("[yellow]No conversation history.[/yellow]")
            return
        
        self.console.print("[bold]📜 Conversation History:[/bold]")
        for i, entry in enumerate(self.conversation_history[-5:], 1):  # Show last 5
            self.console.print(f"\n{i}. [bold cyan]Q:[/bold cyan] {entry['question'][:60]}...")
            self.console.print(f"   [bold green]A:[/bold green] {entry['answer'][:100]}...")
    
    def _export_results(self):
        """Export analysis results."""
        if not self.analysis_results:
            self.console.print("[yellow]No analysis results to export.[/yellow]")
            return
        
        # Create export filename
        timestamp = int(time.time())
        filename = f"analysis_results_{timestamp}.json"
        
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(self.analysis_results, f, indent=2)
            
            self.console.print(f"[green]Results exported to: {filename}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error exporting: {e}[/red]")
    
    def run(self):
        """Run the interactive CLI."""
        # Initialize
        if not self.initialize():
            return
        
        # Show welcome
        self.show_welcome()
        
        # Main loop
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]Ask me about your code[/bold cyan]")
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith('/') or user_input in ['help', 'analyze', 'security', 'performance', 'complexity', 'documentation', 'testing', 'summary', 'priorities', 'recommendations', 'clear', 'history', 'export', 'quit', 'exit']:
                    if not self.handle_command(user_input):
                        break
                else:
                    # Regular question
                    answer = self.ask_question(user_input)
                    self._display_answer("AI Response", answer)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")


def run_interactive_cli(codebase_path: str):
    """Run the enhanced interactive CLI."""
    cli = InteractiveCLI(codebase_path)
    cli.run()
