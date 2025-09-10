"""
GitHub repository analyzer for the Code Quality Intelligence Agent.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests
import zipfile
from urllib.parse import urlparse
import subprocess
import json

from .analyzer import CodeAnalyzer
from .config import config


class GitHubAnalyzer:
    """Analyzes GitHub repositories for code quality issues."""
    
    def __init__(self):
        self.temp_dir = None
        self.repo_info = {}
        self.session = requests.Session()
        
        # Add GitHub API token if available
        from .config import config
        if config.github.api_token:
            self.session.headers.update({
                'Authorization': f'token {config.github.api_token}',
                'Accept': 'application/vnd.github.v3+json'
            })
    
    def analyze_repository(self, repo_url: str, branch: str = "main") -> Dict[str, Any]:
        """Analyze a GitHub repository."""
        try:
            # Parse repository URL
            repo_info = self._parse_github_url(repo_url)
            if not repo_info:
                return {"error": "Invalid GitHub repository URL"}
            
            # Download repository
            repo_path = self._download_repository(repo_info, branch)
            if not repo_path:
                return {"error": "Failed to download repository"}
            
            # Check what files we have
            import os
            all_files = []
            for root, dirs, files in os.walk(repo_path):
                for file in files:
                    if file.endswith(('.py', '.js', '.ts')):
                        all_files.append(os.path.join(root, file))
            
            print(f"📁 Found {len(all_files)} supported files to analyze")
            if len(all_files) > 100:
                print(f"⚠️ Large repository detected ({len(all_files)} files). Analysis may take a while...")
            
            # Analyze the code
            print(f"🔍 Starting analysis of downloaded repository...")
            analyzer = CodeAnalyzer(enhanced_mode=True)
            results = analyzer.analyze_path(repo_path)
            print(f"✅ Analysis completed successfully")
            
            # Add repository metadata
            results["repository"] = {
                "url": repo_url,
                "owner": repo_info["owner"],
                "name": repo_info["name"],
                "branch": branch,
                "local_path": repo_path
            }
            
            return results
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _parse_github_url(self, url: str) -> Optional[Dict[str, str]]:
        """Parse GitHub repository URL."""
        # Handle different GitHub URL formats
        if "github.com" not in url:
            return None
        
        # Remove .git suffix if present
        url = url.replace(".git", "")
        
        # Parse URL
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        
        if len(path_parts) < 2:
            return None
        
        owner = path_parts[0]
        name = path_parts[1]
        
        return {
            "owner": owner,
            "name": name,
            "url": url
        }
    
    def _download_repository(self, repo_info: Dict[str, str], branch: str) -> Optional[str]:
        """Download repository from GitHub."""
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="github_analysis_")
            
            # Try to clone with git first (faster and preserves history)
            if self._try_git_clone(repo_info, branch):
                return self.temp_dir
            
            # Fallback to ZIP download
            return self._download_zip(repo_info, branch)
            
        except Exception as e:
            print(f"Error downloading repository: {e}")
            return None
    
    def _try_git_clone(self, repo_info: Dict[str, str], branch: str) -> bool:
        """Try to clone repository using git."""
        try:
            # Check if git is available
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            
            # Clone repository
            repo_url = f"https://github.com/{repo_info['owner']}/{repo_info['name']}.git"
            result = subprocess.run([
                "git", "clone", "--depth", "1", "--branch", branch, repo_url, self.temp_dir
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully cloned repository using git")
                return True
            else:
                print(f"⚠️ Git clone failed: {result.stderr}")
                return False
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ Git not available, falling back to ZIP download")
            return False
    
    def _download_zip(self, repo_info: Dict[str, str], branch: str) -> Optional[str]:
        """Download repository as ZIP file."""
        try:
            # GitHub API URL for ZIP download
            zip_url = f"https://codeload.github.com/{repo_info['owner']}/{repo_info['name']}/zip/refs/heads/{branch}"
            
            print(f"📥 Downloading repository ZIP from: {zip_url}")
            
            # Download ZIP file
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()
            
            # Save ZIP file
            zip_path = os.path.join(self.temp_dir, "repo.zip")
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            
            # Find extracted directory
            extracted_dirs = [d for d in os.listdir(self.temp_dir) 
                            if os.path.isdir(os.path.join(self.temp_dir, d)) and d != "repo.zip"]
            
            if extracted_dirs:
                # Move contents to temp_dir root
                extracted_path = os.path.join(self.temp_dir, extracted_dirs[0])
                for item in os.listdir(extracted_path):
                    shutil.move(os.path.join(extracted_path, item), self.temp_dir)
                os.rmdir(extracted_path)
            
            # Clean up ZIP file
            os.remove(zip_path)
            
            print(f"✅ Successfully downloaded and extracted repository")
            return self.temp_dir
            
        except Exception as e:
            print(f"❌ ZIP download failed: {e}")
            return None
    
    def _cleanup(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temporary directory")
            except Exception as e:
                print(f"⚠️ Warning: Could not clean up temporary directory: {e}")
    
    def cleanup_after_analysis(self):
        """Clean up temporary directory after analysis is complete."""
        self._cleanup()
    
    def get_repository_info(self, repo_url: str) -> Dict[str, Any]:
        """Get repository information from GitHub API."""
        try:
            repo_info = self._parse_github_url(repo_url)
            if not repo_info:
                return {"error": "Invalid GitHub repository URL"}
            
            # GitHub API URL
            api_url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['name']}"
            
            # Make API request
            response = self.session.get(api_url)
            
            if response.status_code == 403:
                return {"error": "GitHub API rate limit exceeded. Please add a GitHub API token to your environment variables (GITHUB_API_TOKEN) for higher rate limits."}
            
            response.raise_for_status()
            
            repo_data = response.json()
            
            return {
                "name": repo_data.get("name"),
                "full_name": repo_data.get("full_name"),
                "description": repo_data.get("description"),
                "language": repo_data.get("language"),
                "languages": repo_data.get("languages_url"),
                "stars": repo_data.get("stargazers_count", 0),
                "forks": repo_data.get("forks_count", 0),
                "size": repo_data.get("size", 0),
                "created_at": repo_data.get("created_at"),
                "updated_at": repo_data.get("updated_at"),
                "default_branch": repo_data.get("default_branch", "main"),
                "topics": repo_data.get("topics", []),
                "license": repo_data.get("license", {}).get("name") if repo_data.get("license") else None,
                "url": repo_data.get("html_url"),
                "clone_url": repo_data.get("clone_url")
            }
            
        except Exception as e:
            return {"error": f"Failed to get repository info: {str(e)}"}
    
    def list_repository_files(self, repo_url: str, max_files: int = 100) -> List[Dict[str, Any]]:
        """List files in a GitHub repository."""
        try:
            repo_info = self._parse_github_url(repo_url)
            if not repo_info:
                return []
            
            # Get repository contents from GitHub API
            api_url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['name']}/contents"
            response = self.session.get(api_url)
            
            if response.status_code == 403:
                print("⚠️ GitHub API rate limit exceeded. Please add GITHUB_API_TOKEN to your environment variables.")
                return []
            
            if response.status_code != 200:
                return []
            
            files = []
            contents = response.json()
            
            def process_contents(items, path=""):
                for item in items:
                    if len(files) >= max_files:
                        break
                    
                    if item["type"] == "file":
                        # Only include code files
                        file_ext = Path(item["name"]).suffix.lower()
                        if file_ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.txt']:
                            files.append({
                                "path": item["path"],
                                "name": item["name"],
                                "size": item["size"],
                                "download_url": item["download_url"]
                            })
                    elif item["type"] == "dir":
                        # Recursively get files from subdirectories
                        subdir_url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['name']}/contents/{item['path']}"
                        subdir_response = self.session.get(subdir_url)
                        if subdir_response.status_code == 200:
                            process_contents(subdir_response.json(), item["path"])
            
            process_contents(contents)
            print(f"🔍 Found {len(files)} files in repository")
            for file in files[:10]:  # Show first 10 files
                print(f"   📄 {file['path']} ({file['size']} bytes)")
            return files[:max_files]
            
        except Exception as e:
            print(f"Error listing repository files: {e}")
            return []
    
    def analyze_repository_files(self, repo_url: str, selected_files: List[str]) -> Dict[str, Any]:
        """Analyze selected files from a GitHub repository."""
        try:
            repo_info = self._parse_github_url(repo_url)
            if not repo_info:
                return {"error": "Invalid GitHub repository URL"}
            
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="github_analysis_")
            
            # Download and analyze each selected file
            analyzer = CodeAnalyzer()
            all_results = {
                "files_analyzed": [],
                "total_issues": 0,
                "issues_by_category": {},
                "issues_by_severity": {},
                "repository_info": self.get_repository_info(repo_url),
                "summary": {
                    "files_analyzed": 0,
                    "lines_of_code": 0,
                    "total_issues": 0,
                    "issues_by_severity": {},
                    "issues_by_category": {}
                },
                "categories": {},
                "issues": []  # Flat list of all issues
            }
            
            for file_path in selected_files:
                try:
                    # Download file content
                    file_url = f"https://raw.githubusercontent.com/{repo_info['owner']}/{repo_info['name']}/main/{file_path}"
                    response = self.session.get(file_url)
                    
                    if response.status_code != 200:
                        continue
                    
                    # Save file to temp directory
                    local_file_path = os.path.join(self.temp_dir, os.path.basename(file_path))
                    with open(local_file_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    
                    # Analyze file
                    file_results = analyzer.analyze_file(local_file_path)
                    
                    # Add file path to results
                    file_results["file_path"] = file_path
                    all_results["files_analyzed"].append(file_results)
                    
                    # Update summary
                    all_results["summary"]["files_analyzed"] += 1
                    all_results["summary"]["lines_of_code"] += file_results.get("lines_of_code", 0)
                    
                    # Aggregate issues
                    file_issues = len(file_results.get("issues", []))
                    all_results["total_issues"] += file_issues
                    all_results["summary"]["total_issues"] += file_issues
                    
                    for issue in file_results.get("issues", []):
                        category = issue.get("category", "Other")
                        severity = issue.get("severity", "Medium")
                        
                        # Add issue to flat list
                        issue_with_file = issue.copy()
                        issue_with_file["file_path"] = file_path
                        all_results["issues"].append(issue_with_file)
                        
                        # Update category structure for template
                        if category not in all_results["categories"]:
                            all_results["categories"][category] = {"issues": []}
                        all_results["categories"][category]["issues"].append(issue_with_file)
                        
                        # Update counters
                        if category not in all_results["issues_by_category"]:
                            all_results["issues_by_category"][category] = 0
                        all_results["issues_by_category"][category] += 1
                        
                        if severity not in all_results["issues_by_severity"]:
                            all_results["issues_by_severity"][severity] = 0
                        all_results["issues_by_severity"][severity] += 1
                        
                        # Update summary
                        if category not in all_results["summary"]["issues_by_category"]:
                            all_results["summary"]["issues_by_category"][category] = 0
                        all_results["summary"]["issues_by_category"][category] += 1
                        
                        if severity not in all_results["summary"]["issues_by_severity"]:
                            all_results["summary"]["issues_by_severity"][severity] = 0
                        all_results["summary"]["issues_by_severity"][severity] += 1
                    
                except Exception as e:
                    print(f"Error analyzing file {file_path}: {e}")
                    continue
            
            return all_results
            
        except Exception as e:
            return {"error": f"Error analyzing repository files: {str(e)}"}
    
    def analyze_repository_with_info(self, repo_url: str, branch: Optional[str] = None) -> Dict[str, Any]:
        """Analyze repository with additional GitHub metadata."""
        try:
            # Get repository information
            repo_info = self.get_repository_info(repo_url)
            if "error" in repo_info:
                return repo_info
            
            # Use default branch if not specified
            if not branch:
                branch = repo_info.get("default_branch", "main")
            
            # Analyze repository
            analysis_results = self.analyze_repository(repo_url, branch)
            if "error" in analysis_results:
                return analysis_results
            
            # Combine results
            combined_results = {
                **analysis_results,
                "github_info": repo_info,
                "analysis_metadata": {
                    "analyzed_branch": branch,
                    "analysis_timestamp": str(Path().cwd()),
                    "analyzer_version": "1.0.0"
                }
            }
            
            return combined_results
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def list_supported_repositories(self, query: str = "python") -> List[Dict[str, Any]]:
        """List popular repositories matching a query."""
        try:
            # GitHub API search URL
            api_url = f"https://api.github.com/search/repositories"
            params = {
                "q": f"{query} language:python",
                "sort": "stars",
                "order": "desc",
                "per_page": 10
            }
            
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            
            search_results = response.json()
            repositories = []
            
            for repo in search_results.get("items", []):
                repositories.append({
                    "name": repo.get("name"),
                    "full_name": repo.get("full_name"),
                    "description": repo.get("description"),
                    "stars": repo.get("stargazers_count", 0),
                    "language": repo.get("language"),
                    "url": repo.get("html_url"),
                    "clone_url": repo.get("clone_url")
                })
            
            return repositories
            
        except Exception as e:
            return [{"error": f"Failed to search repositories: {str(e)}"}]
