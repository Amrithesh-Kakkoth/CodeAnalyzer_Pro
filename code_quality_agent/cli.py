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
    
    # Check for API key first
    has_api_key = False
    if config.ai.llm_provider == "groq" and config.ai.groq_api_key:
        has_api_key = True
    elif config.ai.llm_provider == "openai" and config.ai.openai_api_key:
        has_api_key = True
    
    if not has_api_key:
        console.print("[yellow]⚠️  No API key found for AI features[/yellow]")
        console.print("[yellow]The chat command requires an API key to work.[/yellow]")
        console.print("[yellow]Please set one of the following environment variables:[/yellow]")
        console.print("[cyan]  GROQ_API_KEY=your_groq_api_key_here[/cyan]")
        console.print("[cyan]  OPENAI_API_KEY=your_openai_key_here[/cyan]")
        console.print("\n[yellow]You can still use the analyze command for basic code analysis:[/yellow]")
        console.print("[cyan]  python -m code_quality_agent analyze .[/cyan]")
        raise click.Abort()
    
    # Use enhanced interactive CLI
    from .interactive_cli import run_interactive_cli
    run_interactive_cli(path)


@cli.command()
@click.argument('path')
@click.option('--query', '-q', help='Debug specific query')
def debug(path, query):
    """Debug RAG system and show indexing information."""
    try:
        from .enhanced_rag_system import EnhancedCodeRAGSystem
        
        console.print(f"[yellow]🔍 Debugging RAG system for: {path}[/yellow]")
        
        # Initialize enhanced RAG system
        enhanced_rag = EnhancedCodeRAGSystem(path)
        
        # Index the codebase
        console.print("[blue]📊 Indexing codebase...[/blue]")
        result = enhanced_rag.index_codebase()
        
        if "error" in result:
            console.print(f"[red]❌ Error: {result['error']}[/red]")
            return
        
        console.print(f"[green]✅ Indexing complete![/green]")
        console.print(f"📄 Documents indexed: {result.get('documents_indexed', 0)}")
        console.print(f"🧩 Entities found: {result.get('entities_found', 0)}")
        console.print(f"🔗 Relationships found: {result.get('relationships_found', 0)}")
        
        # Show debug information
        debug_info = enhanced_rag.debug_vector_store()
        
        if "error" in debug_info:
            console.print(f"[red]❌ Debug error: {debug_info['error']}[/red]")
            return
        
        # Create debug table
        table = Table(title="RAG System Debug Information")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Total Documents", str(debug_info['total_documents']))
        
        # Files breakdown
        files_text = "\n".join([f"{filename}: {count}" for filename, count in debug_info['files'].items()])
        table.add_row("Files", files_text[:100] + "..." if len(files_text) > 100 else files_text)
        
        # Entity types breakdown
        entity_types_text = "\n".join([f"{entity_type}: {count}" for entity_type, count in debug_info['entity_types'].items()])
        table.add_row("Entity Types", entity_types_text[:100] + "..." if len(entity_types_text) > 100 else entity_types_text)
        
        # Languages breakdown
        languages_text = "\n".join([f"{language}: {count}" for language, count in debug_info['languages'].items()])
        table.add_row("Languages", languages_text[:100] + "..." if len(languages_text) > 100 else languages_text)
        
        console.print(table)
        
        # Debug specific query if provided
        if query:
            console.print(f"\n[blue]🔍 Debugging query: '{query}'[/blue]")
            search_debug = enhanced_rag.search_debug(query)
            
            query_table = Table(title=f"Query Debug: '{query}'")
            query_table.add_column("Field", style="cyan")
            query_table.add_column("Value", style="yellow")
            
            query_table.add_row("Filename Detected", str(search_debug['filename_detected'] or 'None'))
            query_table.add_row("File Search Results", str(len(search_debug['file_search_results'])))
            query_table.add_row("Similarity Search Results", str(len(search_debug['similarity_search_results'])))
            query_table.add_row("Total Results", str(search_debug['total_results']))
            
            console.print(query_table)
            
            # Show detailed results
            if search_debug['file_search_results']:
                console.print("\n[green]📁 File Search Results:[/green]")
                for i, result in enumerate(search_debug['file_search_results'][:3], 1):
                    console.print(f"{i}. {result['filename']}:{result['line_number']} ({result['type']})")
                    console.print(f"   Entity: {result['entity_name']}")
                    console.print(f"   Preview: {result['content_preview']}")
            
            if search_debug['similarity_search_results']:
                console.print("\n[blue]🔍 Similarity Search Results:[/blue]")
                for i, result in enumerate(search_debug['similarity_search_results'][:3], 1):
                    console.print(f"{i}. {result['filename']}:{result['line_number']} ({result['type']})")
                    console.print(f"   Entity: {result['entity_name']}")
                    console.print(f"   Preview: {result['content_preview']}")
        
    except Exception as e:
        console.print(f"[red]❌ Debug failed: {e}[/red]")


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
