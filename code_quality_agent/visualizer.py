"""
Visualization and insights generation for code analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import networkx as nx
from datetime import datetime

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CodeVisualizer:
    """Generates visualizations and insights from code analysis."""
    
    def __init__(self, output_dir: str = "./visualizations"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Color schemes
        self.colors = {
            'critical': '#FF4444',
            'high': '#FF8800',
            'medium': '#FFBB00',
            'low': '#00BB00',
            'info': '#0088FF'
        }
        
        self.severity_colors = {
            'critical': '#DC2626',
            'high': '#EA580C',
            'medium': '#D97706',
            'low': '#16A34A',
            'info': '#2563EB'
        }
    
    def generate_quality_dashboard(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate a comprehensive quality dashboard."""
        dashboard_files = {}
        
        # Generate individual visualizations
        dashboard_files['issues_by_category'] = self._plot_issues_by_category(analysis_results)
        dashboard_files['severity_distribution'] = self._plot_severity_distribution(analysis_results)
        dashboard_files['files_complexity'] = self._plot_files_complexity(analysis_results)
        dashboard_files['priority_matrix'] = self._plot_priority_matrix(analysis_results)
        dashboard_files['quality_trends'] = self._plot_quality_trends(analysis_results)
        
        # Generate summary report
        dashboard_files['summary_report'] = self._generate_summary_report(analysis_results)
        
        return dashboard_files
    
    def _plot_issues_by_category(self, results: Dict[str, Any]) -> str:
        """Plot issues by category."""
        categories = results.get('categories', {})
        
        if not categories:
            return self._create_empty_plot("No Issues Found", "No issues detected in the analysis")
        
        # Prepare data
        category_names = []
        issue_counts = []
        colors = []
        
        for category, data in categories.items():
            category_names.append(category.title())
            count = len(data.get('issues', []))
            issue_counts.append(count)
            
            # Color based on severity
            if count > 10:
                colors.append(self.colors['critical'])
            elif count > 5:
                colors.append(self.colors['high'])
            elif count > 2:
                colors.append(self.colors['medium'])
            else:
                colors.append(self.colors['low'])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(category_names, issue_counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, issue_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Issues by Category', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Number of Issues', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if needed
        if len(category_names) > 4:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        filename = self.output_dir / 'issues_by_category.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_severity_distribution(self, results: Dict[str, Any]) -> str:
        """Plot severity distribution."""
        categories = results.get('categories', {})
        
        # Count issues by severity
        severity_counts = Counter()
        for category_data in categories.values():
            for issue in category_data.get('issues', []):
                severity = issue.get('severity', 'low')
                severity_counts[severity] += 1
        
        if not severity_counts:
            return self._create_empty_plot("No Issues Found", "No issues detected in the analysis")
        
        # Prepare data
        severities = list(severity_counts.keys())
        counts = list(severity_counts.values())
        colors = [self.severity_colors.get(sev, self.colors['info']) for sev in severities]
        
        # Create pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Pie chart
        wedges, texts, autotexts = ax1.pie(counts, labels=severities, colors=colors, autopct='%1.1f%%',
                                          startangle=90, explode=[0.05] * len(severities))
        
        # Enhance pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax1.set_title('Severity Distribution (Pie Chart)', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(severities, counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('Severity Distribution (Bar Chart)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Severity Level')
        ax2.set_ylabel('Number of Issues')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = self.output_dir / 'severity_distribution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_files_complexity(self, results: Dict[str, Any]) -> str:
        """Plot files complexity analysis."""
        categories = results.get('categories', {})
        
        # Extract file complexity data
        file_complexity = defaultdict(int)
        file_issues = defaultdict(list)
        
        for category_data in categories.values():
            for issue in category_data.get('issues', []):
                file_path = issue.get('file', 'unknown')
                file_name = Path(file_path).name
                
                # Calculate complexity score based on issue severity
                severity_scores = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'info': 0}
                score = severity_scores.get(issue.get('severity', 'low'), 1)
                
                file_complexity[file_name] += score
                file_issues[file_name].append(issue)
        
        if not file_complexity:
            return self._create_empty_plot("No Files Analyzed", "No files were analyzed")
        
        # Sort files by complexity
        sorted_files = sorted(file_complexity.items(), key=lambda x: x[1], reverse=True)
        top_files = sorted_files[:10]  # Top 10 most complex files
        
        file_names = [item[0] for item in top_files]
        complexity_scores = [item[1] for item in top_files]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color bars based on complexity
        colors = []
        for score in complexity_scores:
            if score > 15:
                colors.append(self.colors['critical'])
            elif score > 10:
                colors.append(self.colors['high'])
            elif score > 5:
                colors.append(self.colors['medium'])
            else:
                colors.append(self.colors['low'])
        
        bars = ax.barh(file_names, complexity_scores, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, score in zip(bars, complexity_scores):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                   f'{score}', ha='left', va='center', fontweight='bold')
        
        ax.set_title('File Complexity Analysis (Top 10)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Complexity Score', fontsize=12)
        ax.set_ylabel('Files', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save plot
        filename = self.output_dir / 'files_complexity.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_priority_matrix(self, results: Dict[str, Any]) -> str:
        """Plot priority matrix for issues."""
        categories = results.get('categories', {})
        
        # Extract priority data (if available)
        priority_data = defaultdict(lambda: defaultdict(int))
        
        for category_data in categories.values():
            for issue in category_data.get('issues', []):
                category = issue.get('category', 'general')
                priority = issue.get('priority', 'P3 - Low')
                
                # Extract priority level
                if priority.startswith('P0'):
                    priority_level = 'P0 - Critical'
                elif priority.startswith('P1'):
                    priority_level = 'P1 - High'
                elif priority.startswith('P2'):
                    priority_level = 'P2 - Medium'
                elif priority.startswith('P3'):
                    priority_level = 'P3 - Low'
                else:
                    priority_level = 'P4 - Info'
                
                priority_data[category][priority_level] += 1
        
        if not priority_data:
            return self._create_empty_plot("No Priority Data", "No priority information available")
        
        # Create heatmap
        categories = list(priority_data.keys())
        priorities = ['P0 - Critical', 'P1 - High', 'P2 - Medium', 'P3 - Low', 'P4 - Info']
        
        # Create matrix
        matrix = []
        for category in categories:
            row = []
            for priority in priorities:
                row.append(priority_data[category][priority])
            matrix.append(row)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='Reds', aspect='auto')
        
        # Set ticks
        ax.set_xticks(range(len(priorities)))
        ax.set_yticks(range(len(categories)))
        ax.set_xticklabels(priorities, rotation=45, ha='right')
        ax.set_yticklabels(categories)
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(priorities)):
                text = ax.text(j, i, matrix[i][j], ha="center", va="center", 
                             color="white" if matrix[i][j] > max(max(row) for row in matrix) / 2 else "black",
                             fontweight='bold')
        
        ax.set_title('Priority Matrix by Category', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Priority Level', fontsize=12)
        ax.set_ylabel('Category', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Issues', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        filename = self.output_dir / 'priority_matrix.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_quality_trends(self, results: Dict[str, Any]) -> str:
        """Plot quality trends (placeholder for future time-series data)."""
        # This would be used for tracking quality over time
        # For now, create a placeholder visualization
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Simulate trend data
        dates = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
        quality_scores = [85, 82, 88, 91]  # Simulated quality scores
        
        ax.plot(dates, quality_scores, marker='o', linewidth=3, markersize=8, 
                color=self.colors['medium'], alpha=0.8)
        
        # Fill area under curve
        ax.fill_between(dates, quality_scores, alpha=0.3, color=self.colors['medium'])
        
        ax.set_title('Code Quality Trends (Simulated)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Quality Score', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for date, score in zip(dates, quality_scores):
            ax.annotate(f'{score}%', (date, score), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        filename = self.output_dir / 'quality_trends.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _create_empty_plot(self, title: str, message: str) -> str:
        """Create an empty plot with a message."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14, 
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        filename = self.output_dir / f'{title.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a text summary report."""
        summary = results.get('summary', {})
        categories = results.get('categories', {})
        
        # Calculate statistics
        total_issues = summary.get('total_issues', 0)
        files_analyzed = summary.get('files_analyzed', 0)
        lines_of_code = summary.get('lines_of_code', 0)
        
        # Count by severity
        severity_counts = Counter()
        for category_data in categories.values():
            for issue in category_data.get('issues', []):
                severity = issue.get('severity', 'low')
                severity_counts[severity] += 1
        
        # Generate report
        report_lines = [
            "# Code Quality Analysis Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- **Files Analyzed**: {files_analyzed}",
            f"- **Lines of Code**: {lines_of_code:,}",
            f"- **Total Issues**: {total_issues}",
            "",
            "## Issues by Severity",
        ]
        
        for severity in ['critical', 'high', 'medium', 'low']:
            count = severity_counts.get(severity, 0)
            report_lines.append(f"- **{severity.title()}**: {count}")
        
        report_lines.extend([
            "",
            "## Issues by Category",
        ])
        
        for category, data in categories.items():
            count = len(data.get('issues', []))
            report_lines.append(f"- **{category.title()}**: {count}")
        
        report_lines.extend([
            "",
            "## Recommendations",
            "- Focus on critical and high-priority issues first",
            "- Implement automated code quality checks",
            "- Consider code review processes for complex files",
            "- Regular monitoring of code quality metrics"
        ])
        
        # Save report
        filename = self.output_dir / 'summary_report.md'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        return str(filename)
    
    def generate_dependency_graph(self, codebase_path: str) -> str:
        """Generate dependency graph visualization."""
        try:
            # This is a simplified version - in practice, you'd parse actual imports
            G = nx.DiGraph()
            
            # Add some sample nodes and edges
            G.add_node("main.py", type="module")
            G.add_node("utils.py", type="module")
            G.add_node("config.py", type="config")
            G.add_node("models.py", type="models")
            
            G.add_edge("main.py", "utils.py")
            G.add_edge("main.py", "config.py")
            G.add_edge("utils.py", "models.py")
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Layout
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Draw nodes
            node_colors = []
            for node in G.nodes():
                node_type = G.nodes[node].get('type', 'module')
                if node_type == 'config':
                    node_colors.append(self.colors['medium'])
                elif node_type == 'models':
                    node_colors.append(self.colors['low'])
                else:
                    node_colors.append(self.colors['info'])
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                 node_size=2000, alpha=0.8, ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                 arrows=True, arrowsize=20, ax=ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
            
            ax.set_title('Code Dependency Graph', fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')
            
            plt.tight_layout()
            
            # Save plot
            filename = self.output_dir / 'dependency_graph.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filename)
            
        except Exception as e:
            print(f"Error generating dependency graph: {e}")
            return self._create_empty_plot("Dependency Graph Error", f"Could not generate dependency graph: {e}")
    
    def create_html_dashboard(self, analysis_results: Dict[str, Any]) -> str:
        """Create an HTML dashboard with all visualizations."""
        dashboard_files = self.generate_quality_dashboard(analysis_results)
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Code Quality Dashboard</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .chart {{
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-radius: 8px;
                    border: 1px solid #ddd;
                }}
                .chart img {{
                    width: 100%;
                    height: auto;
                    border-radius: 5px;
                }}
                .summary {{
                    background-color: #e8f4f8;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #2196F3;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Code Quality Intelligence Dashboard</h1>
                
                <div class="summary">
                    <h2>Analysis Summary</h2>
                    <p><strong>Files Analyzed:</strong> {analysis_results.get('summary', {}).get('files_analyzed', 0)}</p>
                    <p><strong>Lines of Code:</strong> {analysis_results.get('summary', {}).get('lines_of_code', 0):,}</p>
                    <p><strong>Total Issues:</strong> {analysis_results.get('summary', {}).get('total_issues', 0)}</p>
                </div>
                
                <div class="grid">
                    <div class="chart">
                        <h3>Issues by Category</h3>
                        <img src="{Path(dashboard_files['issues_by_category']).name}" alt="Issues by Category">
                    </div>
                    
                    <div class="chart">
                        <h3>Severity Distribution</h3>
                        <img src="{Path(dashboard_files['severity_distribution']).name}" alt="Severity Distribution">
                    </div>
                    
                    <div class="chart">
                        <h3>File Complexity Analysis</h3>
                        <img src="{Path(dashboard_files['files_complexity']).name}" alt="File Complexity">
                    </div>
                    
                    <div class="chart">
                        <h3>Priority Matrix</h3>
                        <img src="{Path(dashboard_files['priority_matrix']).name}" alt="Priority Matrix">
                    </div>
                    
                    <div class="chart">
                        <h3>Quality Trends</h3>
                        <img src="{Path(dashboard_files['quality_trends']).name}" alt="Quality Trends">
                    </div>
                    
                    <div class="chart">
                        <h3>Dependency Graph</h3>
                        <img src="{Path(dashboard_files.get('dependency_graph', '')).name}" alt="Dependency Graph">
                    </div>
                </div>
                
                <div class="summary">
                    <h2>📊 Detailed Report</h2>
                    <p>For detailed analysis, see: <a href="{Path(dashboard_files['summary_report']).name}">Summary Report</a></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML dashboard
        filename = self.output_dir / 'dashboard.html'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(filename)

