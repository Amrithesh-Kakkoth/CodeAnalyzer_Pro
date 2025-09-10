"""
Command-line interface for the Code Quality Intelligence Agent.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from .analyzer import CodeAnalyzer
from .qa_agent import CodeQAAgent
from .config import config

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Code Quality Intelligence Agent - AI-powered code analysis tool."""
    pass


@cli.command()
@click.argument('path', type=click.Path())
@click.option('--language', '-l', 
              type=click.Choice(['python', 'javascript', 'typescript', 'auto']),
              default='auto', help='Programming language to analyze')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--format', '-f', 
              type=click.Choice(['json', 'html', 'markdown']),
              default='json', help='Output format')
@click.option('--enhanced', is_flag=True, help='Use enhanced analysis mode')
def analyze(path, language, output, format, enhanced):
    """Analyze code quality for a file or directory."""
    
    # Check if it's a GitHub URL first
    is_github_repo = path.startswith(('http://github.com/', 'https://github.com/'))
    
    # Check if path exists (only for local paths)
    if not is_github_repo and not Path(path).exists():
        console.print(f"[red]Error: Path '{path}' does not exist.[/red]")
        raise click.Abort()
    
    if is_github_repo:
        # Handle GitHub repository analysis
        console.print(f"[yellow]GitHub repository detected: {path}[/yellow]")
        
        # Extract branch if specified
        branch = "main"
        if '@' in path:
            path, branch = path.split('@')
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Downloading and analyzing GitHub repository...", total=None)
            
            try:
                from .github_analyzer import GitHubAnalyzer
                github_analyzer = GitHubAnalyzer()
                results = github_analyzer.analyze_repository_with_info(path, branch)
                
                if "error" in results:
                    console.print(f"[red]Error: {results['error']}[/red]")
                    raise click.Abort()
                
                progress.update(task, description="GitHub analysis complete!")
                
                # Display GitHub info
                if "github_info" in results:
                    _display_github_info(results["github_info"])
                
                # Display analysis results
                _display_analysis_results(results)
                
                # Cleanup
                github_analyzer.cleanup_after_analysis()
                
            except Exception as e:
                console.print(f"[red]Error analyzing GitHub repository: {e}[/red]")
                raise click.Abort()
    else:
        # Handle local file/directory analysis
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing code...", total=None)
            
            try:
                analyzer = CodeAnalyzer(enhanced_mode=enhanced)
                results = analyzer.analyze_path(path)
                
                progress.update(task, description="Analysis complete!")
                
                # Display results
                _display_analysis_results(results)
                
                # Save to file if specified
                if output:
                    _save_results(results, output, format)
                
            except Exception as e:
                console.print(f"[red]Error analyzing code: {e}[/red]")
                raise click.Abort()


@cli.command()
@click.argument('path')
@click.option('--model', '-m', default='auto', help='LLM model to use for Q&A')
@click.option('--enhanced', is_flag=True, help='Use enhanced interactive CLI')
def chat(path, model, enhanced):
    """Start interactive Q&A session about the codebase."""
    # Check if it's a GitHub URL
    is_github_repo = path.startswith(('http://github.com/', 'https://github.com/'))
    
    # Check if path exists (only for local paths)
    if not is_github_repo and not Path(path).exists():
        console.print(f"[red]Error: Path '{path}' does not exist.[/red]")
        raise click.Abort()
    
    if enhanced:
        # Use enhanced interactive CLI
        from .interactive_cli import run_interactive_cli
        run_interactive_cli(path)
    else:
        # Original simple chat
        # Check for API key based on provider
        if config.ai.llm_provider == "groq" and not config.ai.groq_api_key:
            console.print("[red]Error: Groq API key not found. Please set GROQ_API_KEY environment variable.[/red]")
            raise click.Abort()
        elif config.ai.llm_provider == "openai" and not config.ai.openai_api_key:
            console.print("[red]Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.[/red]")
            raise click.Abort()
        
        # Display header
        console.print(Panel.fit(
            f"[bold blue]Code Quality Intelligence Agent - Q&A Mode[/bold blue]\n"
            f"Repository: [green]{path}[/green]\n"
            f"Model: [yellow]{config.ai.groq_model_name if config.ai.llm_provider == 'groq' else config.ai.openai_model_name}[/yellow]\n\n"
            f"Type your questions about the codebase. Type 'quit' to exit.",
            title="Interactive Q&A"
        ))
        
        try:
            # Initialize Q&A agent
            qa_agent = CodeQAAgent(path)
            
            # Start interactive session
            while True:
                try:
                    question = click.prompt("\n[bold cyan]Your question[/bold cyan]", type=str)
                    
                    if question.lower() in ['quit', 'exit', 'q']:
                        console.print("[yellow]Goodbye![/yellow]")
                        break
                    
                    if not question.strip():
                        continue
                    
                    # Get answer from AI agent
                    with console.status("[bold green]Thinking..."):
                        answer = qa_agent.ask_question(question)
                    
                    console.print(f"\n[bold green]Answer:[/bold green]\n{answer}")
                    
                except KeyboardInterrupt:
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    
        except Exception as e:
            console.print(f"[red]Error initializing Q&A agent: {e}[/red]")
            raise click.Abort()


@cli.command()
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Number of repositories to return')
def search(query, limit):
    """Search for popular GitHub repositories."""
    try:
        from .github_analyzer import GitHubAnalyzer
        github_analyzer = GitHubAnalyzer()
        
        console.print(f"[yellow]Searching for repositories matching '{query}'...[/yellow]")
        
        repos = github_analyzer.list_supported_repositories(query, limit)
        
        if not repos:
            console.print("[red]No repositories found.[/red]")
            return
        
        # Display results
        table = Table(title=f"GitHub Repositories: {query}")
        table.add_column("Repository", style="cyan")
        table.add_column("Stars", style="yellow")
        table.add_column("Language", style="green")
        table.add_column("Description", style="white")
        
        for repo in repos:
            table.add_row(
                repo['full_name'],
                str(repo['stargazers_count']),
                repo['language'] or 'N/A',
                repo['description'][:50] + '...' if repo['description'] and len(repo['description']) > 50 else repo['description'] or 'N/A'
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error searching repositories: {e}[/red]")




def _display_analysis_results(results):
    """Display analysis results in a formatted table."""
    if not results or 'issues' not in results:
        console.print("[yellow]No issues found![/yellow]")
        return
    
    # Create summary table
    table = Table(title="Code Quality Analysis Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="yellow")
    
    total_issues = len(results['issues'])
    high_priority = len([i for i in results['issues'] if i.get('severity') == 'high'])
    medium_priority = len([i for i in results['issues'] if i.get('severity') == 'medium'])
    low_priority = len([i for i in results['issues'] if i.get('severity') == 'low'])
    
    table.add_row("Total Issues", str(total_issues))
    table.add_row("High Priority", str(high_priority))
    table.add_row("Medium Priority", str(medium_priority))
    table.add_row("Low Priority", str(low_priority))
    
    console.print(table)
    
    # Show top issues
    if results['issues']:
        console.print("\n[bold]Top Issues:[/bold]")
        for i, issue in enumerate(results['issues'][:5], 1):
            console.print(f"{i}. [{issue.get('severity', 'unknown').upper()}] {issue.get('message', 'No message')}")


def _display_github_info(github_info):
    """Display GitHub repository information."""
    console.print(f"\n[bold]GitHub Repository Information:[/bold]")
    console.print(f"📁 Repository: [cyan]{github_info.get('full_name', 'N/A')}[/cyan]")
    console.print(f"⭐ Stars: [yellow]{github_info.get('stars', 0):,}[/yellow]")
    console.print(f"🍴 Forks: [blue]{github_info.get('forks', 0):,}[/blue]")
    console.print(f"💻 Language: [green]{github_info.get('language', 'N/A')}[/green]")
    console.print(f"📏 Size: [magenta]{github_info.get('size', 0):,} KB[/magenta]")
    if github_info.get('description'):
        console.print(f"📝 Description: {github_info['description']}")
    if github_info.get('license'):
        console.print(f"📄 License: {github_info['license']}")
    if github_info.get('topics'):
        topics = ", ".join(github_info['topics'][:5])
        console.print(f"🏷️ Topics: {topics}")


def _save_results(results, output_path, format):
    """Save results to a file."""
    try:
        if format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif format == 'html':
            # Basic HTML output
            html = f"<html><body><h1>Code Quality Analysis Results</h1><pre>{results}</pre></body></html>"
            with open(output_path, 'w') as f:
                f.write(html)
        elif format == 'markdown':
            # Basic markdown output
            md = f"# Code Quality Analysis Results\n\n```json\n{results}\n```"
            with open(output_path, 'w') as f:
                f.write(md)
        
        console.print(f"[green]Results saved to: {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error saving results: {e}[/red]")


if __name__ == '__main__':
    cli()
