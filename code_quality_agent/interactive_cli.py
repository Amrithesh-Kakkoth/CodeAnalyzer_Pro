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
        self.actual_repo_path = codebase_path  # Will be updated for GitHub repos
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
            
            # Initialize components with timeout protection
            with self.console.status("[bold green]Initializing AI agent..."):
                try:
                    self.qa_agent = CodeQAAgent(str(self.codebase_path))
                    
                    # For GitHub repos, update the actual path to the downloaded repository
                    if self.is_github_repo and hasattr(self.qa_agent, 'enhanced_rag') and self.qa_agent.enhanced_rag:
                        self.actual_repo_path = str(self.qa_agent.enhanced_rag.codebase_path)
                except Exception as e:
                    self.console.print(f"[yellow]Warning: AI agent initialization failed: {e}[/yellow]")
                    self.console.print("[yellow]Continuing with basic analysis mode...[/yellow]")
                    self.qa_agent = None
                
            with self.console.status("[bold green]Initializing analyzer..."):
                self.analyzer = CodeAnalyzer(enhanced_mode=False)  # Use basic mode to avoid hangs
                try:
                    self.scorer = SeverityScorer()
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Severity scorer initialization failed: {e}[/yellow]")
                    self.scorer = None
                
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error initializing: {e}[/red]")
            return False
    
    def _check_api_key(self) -> bool:
        """Check if API key is configured."""
        has_api_key = False
        if config.ai.llm_provider == "groq" and config.ai.groq_api_key:
            has_api_key = True
        elif config.ai.llm_provider == "openai" and config.ai.openai_api_key:
            has_api_key = True
        
        if not has_api_key:
            self.console.print("[yellow]Warning: No API key found for AI features.[/yellow]")
            self.console.print("[yellow]You can still use basic analysis features.[/yellow]")
            if config.ai.llm_provider == "groq":
                self.console.print("[yellow]To enable AI features, set GROQ_API_KEY environment variable.[/yellow]")
            elif config.ai.llm_provider == "openai":
                self.console.print("[yellow]To enable AI features, set OPENAI_API_KEY environment variable.[/yellow]")
        
        return True  # Always return True to allow basic functionality
    
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
        
        # Create a beautiful welcome layout
        self.console.print("\n")
        
        # Main title
        title = Panel.fit(
            "[bold white]🤖 Code Quality Intelligence Agent[/bold white]",
            border_style="bright_blue",
            padding=(1, 2)
        )
        self.console.print(title)
        
        # Codebase info panel
        info_panel = Panel(
            f"{codebase_info}\n\n[bold]📊 Files to analyze:[/bold] {file_count}\n[bold]🧠 AI Model:[/bold] {config.ai.groq_model_name if config.ai.llm_provider == 'groq' else config.ai.openai_model_name}\n[bold]🔍 Provider:[/bold] {config.ai.llm_provider.upper()}",
            title="[bold cyan]📋 Project Info[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(info_panel)
        
        # Quick actions panel
        actions_text = """[bold green]✨ Quick Actions:[/bold green]
• [bold]analyze[/bold] - Run comprehensive code quality analysis
• [bold]security[/bold] - Check for security vulnerabilities  
• [bold]performance[/bold] - Identify performance bottlenecks
• [bold]complexity[/bold] - Analyze code complexity
• [bold]documentation[/bold] - Review documentation gaps
• [bold]testing[/bold] - Assess testing coverage

[bold yellow]💡 Ask me anything about your code![/bold yellow]
[dim]Examples: "What are the main security risks?", "How can I improve performance?"[/dim]"""
        
        actions_panel = Panel(
            actions_text,
            title="[bold green]🚀 Available Commands[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(actions_panel)
        
        # Help hint
        help_hint = Panel.fit(
            "[dim]Type 'help' for detailed commands • Type 'quit' to exit[/dim]",
            border_style="dim"
        )
        self.console.print(help_hint)
        self.console.print()
    
    def show_help(self):
        """Display professional help information."""
        self.console.print("\n")
        
        # Main help title
        help_title = Panel.fit(
            "[bold white]📚 Command Reference[/bold white]",
            border_style="bright_blue",
            padding=(1, 2)
        )
        self.console.print(help_title)
        
        # Analysis commands
        analysis_text = """[bold cyan]🔍 Analysis Commands:[/bold cyan]
• [bold]analyze[/bold] - Run comprehensive code quality analysis
• [bold]security[/bold] - Focus on security vulnerabilities
• [bold]performance[/bold] - Analyze performance bottlenecks
• [bold]complexity[/bold] - Check code complexity metrics
• [bold]documentation[/bold] - Review documentation gaps
• [bold]testing[/bold] - Identify testing needs and coverage"""
        
        analysis_panel = Panel(
            analysis_text,
            title="[bold cyan]📊 Analysis[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(analysis_panel)
        
        # Interactive commands
        interactive_text = """[bold green]💬 Interactive Commands:[/bold green]
• [bold]summary[/bold] - Get comprehensive codebase overview
• [bold]priorities[/bold] - Show prioritized issues by severity
• [bold]recommendations[/bold] - Get improvement suggestions
• [bold]trends[/bold] - Analyze quality trends over time"""
        
        interactive_panel = Panel(
            interactive_text,
            title="[bold green]🎯 Insights[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(interactive_panel)
        
        # Example questions
        examples_text = """[bold yellow]💡 Example Questions:[/bold yellow]
• "What are the main security risks in this code?"
• "How can I improve the performance of this function?"
• "What's the most complex part of this codebase?"
• "What testing should I add to improve coverage?"
• "How would you refactor this code for better maintainability?"
• "What design patterns are being used here?" """
        
        examples_panel = Panel(
            examples_text,
            title="[bold yellow]❓ Ask Me Anything[/bold yellow]",
            border_style="yellow",
            padding=(1, 2)
        )
        self.console.print(examples_panel)
        
        # Utility commands
        utility_text = """[bold magenta]🛠️ Utility Commands:[/bold magenta]
• [bold]help[/bold] - Show this command reference
• [bold]clear[/bold] - Clear conversation history
• [bold]history[/bold] - Show conversation history
• [bold]export[/bold] - Export analysis results
• [bold]quit[/bold] / [bold]exit[/bold] - Exit the application"""
        
        utility_panel = Panel(
            utility_text,
            title="[bold magenta]⚙️ Utilities[/bold magenta]",
            border_style="magenta",
            padding=(1, 2)
        )
        self.console.print(utility_panel)
        
        # Footer
        footer = Panel.fit(
            "[dim]💡 Tip: You can ask natural language questions or use commands above[/dim]",
            border_style="dim"
        )
        self.console.print(footer)
        self.console.print()
    
    def run_analysis(self) -> bool:
        """Run full code quality analysis."""
        try:
            self.console.print("\n")
            
            # Analysis header
            analysis_header = Panel.fit(
                "[bold white]🔍 Code Quality Analysis[/bold white]",
                border_style="bright_blue",
                padding=(1, 2)
            )
            self.console.print(analysis_header)
            
            with self.console.status("[bold green]Analyzing codebase...", spinner="dots"):
                # Use the actual repository path (downloaded for GitHub repos)
                self.analysis_results = self.analyzer.analyze_path(str(self.actual_repo_path))
            
            # Display results
            self._display_analysis_summary()
            return True
            
        except Exception as e:
            error_panel = Panel(
                f"[bold red]❌ Analysis Error[/bold red]\n\n[red]{str(e)}[/red]",
                title="[bold red]Error[/bold red]",
                border_style="red",
                padding=(1, 2)
            )
            self.console.print(error_panel)
            return False
    
    def _display_analysis_summary(self):
        """Display professional analysis summary."""
        if not self.analysis_results:
            return
        
        summary = self.analysis_results['summary']
        categories = self.analysis_results['categories']
        
        # Create summary table with better styling
        table = Table(
            title="[bold cyan]📊 Analysis Summary[/bold cyan]",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan"
        )
        table.add_column("Metric", style="bold white", min_width=20)
        table.add_column("Value", style="bold yellow", justify="right", min_width=15)
        
        table.add_row("📁 Files Analyzed", f"[bold]{summary['files_analyzed']}[/bold]")
        table.add_row("📝 Lines of Code", f"[bold]{summary['lines_of_code']:,}[/bold]")
        table.add_row("⚠️ Total Issues", f"[bold]{summary['total_issues']}[/bold]")
        
        # Add category breakdown with colors
        for category, data in categories.items():
            count = len(data['issues'])
            if count > 0:
                if category.lower() == 'security':
                    style = "bold red"
                    icon = "🔒"
                elif category.lower() == 'performance':
                    style = "bold yellow"
                    icon = "⚡"
                elif category.lower() == 'complexity':
                    style = "bold magenta"
                    icon = "🧩"
                else:
                    style = "bold blue"
                    icon = "📋"
                table.add_row(f"{icon} {category.title()} Issues", f"[{style}]{count}[/{style}]")
        
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
        """Display AI answer in a professional formatted panel."""
        # Format answer as markdown
        formatted_answer = Markdown(answer)
        
        # Create a beautiful answer panel
        answer_panel = Panel(
            formatted_answer,
            title=f"[bold white]🤖 {title}[/bold white]",
            border_style="bright_green",
            padding=(1, 2),
            expand=False
        )
        
        self.console.print(answer_panel)
        self.console.print()  # Add spacing
    
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
                # Get user input with better styling
                user_input = Prompt.ask(
                    "\n[bold white]💬[/bold white] [bold cyan]Ask me about your code[/bold cyan]",
                    default="",
                    show_default=False
                )
                
                if not user_input.strip():
                    continue
                
                # Handle commands - check for exact matches first
                command = user_input.lower().strip()
                if command == 'analyze':
                    self.console.print("[bold green]🔍 Running Code Quality Analysis...[/bold green]")
                    if not self.run_analysis():
                        self.console.print("[red]Analysis failed. Please try again.[/red]")
                elif command in ['help', 'security', 'performance', 'complexity', 'documentation', 'testing', 'summary', 'priorities', 'recommendations', 'clear', 'history', 'export', 'quit', 'exit', 'q']:
                    if not self.handle_command(command):
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
