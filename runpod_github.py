"""
GitHub Integration for RunPod
=============================
Handle Git operations (clone, commit, push) for model storage.

Supports both SSH keys and HTTPS tokens for authentication.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List


def configure_git_for_large_files(repo_path: Optional[Path] = None) -> None:
    """
    Configure Git settings for pushing large files.
    Increases buffer size and timeout to handle large model files.
    """
    repo_path = repo_path or Path.cwd()
    try:
        # Increase HTTP post buffer for large files (500MB)
        run_git_command(['config', 'http.postBuffer', '524288000'], cwd=repo_path, check=False)
        # Increase HTTP timeout (5 minutes)
        run_git_command(['config', 'http.timeout', '300'], cwd=repo_path, check=False)
        # Enable HTTP version 1.1 (more stable for large uploads)
        run_git_command(['config', 'http.version', 'HTTP/1.1'], cwd=repo_path, check=False)
        # Increase compression level (faster for large files)
        run_git_command(['config', 'core.compression', '0'], cwd=repo_path, check=False)
    except Exception as e:
        print(f"Warning: Could not configure Git for large files: {e}")


def run_git_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> tuple[int, str, str]:
    """
    Run a git command and return the result.
    
    Args:
        cmd: Git command as list of strings
        cwd: Working directory (default: current directory)
        check: If True, raise exception on non-zero exit code
        
    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    try:
        result = subprocess.run(
            ['git'] + cmd,
            cwd=cwd or Path.cwd(),
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        if check:
            raise
        return e.returncode, e.stdout, e.stderr
    except FileNotFoundError:
        raise RuntimeError("Git is not installed. Please install git first.")


def is_git_repo(path: Path) -> bool:
    """Check if a directory is a git repository."""
    return (path / '.git').exists()


def setup_git_config(name: str = "RunPod Bot", email: str = "runpod@bot.local"):
    """
    Setup git config for commits.
    
    Args:
        name: Git user name
        email: Git user email
    """
    run_git_command(['config', 'user.name', name], check=False)
    run_git_command(['config', 'user.email', email], check=False)


def check_git_auth(repo_path: Path) -> bool:
    """
    Check if Git authentication is configured.
    
    Returns:
        True if authentication is available, False otherwise
    """
    # Check for SSH key
    ssh_key_path = Path.home() / '.ssh' / 'id_rsa'
    if ssh_key_path.exists():
        return True
    
    # Check for HTTPS token in environment
    if 'GITHUB_TOKEN' in os.environ:
        return True
    
    # Check if remote URL uses token
    try:
        returncode, stdout, _ = run_git_command(['remote', 'get-url', 'origin'], cwd=repo_path, check=False)
        if returncode == 0:
            url = stdout.strip()
            # Check if URL contains token (format: https://token@github.com/...)
            if '@' in url and 'github.com' in url:
                return True
    except:
        pass
    
    return False


def setup_github_auth(repo_path: Path, token: Optional[str] = None) -> bool:
    """
    Setup GitHub authentication.
    
    Args:
        repo_path: Path to repository
        token: GitHub personal access token (optional, can use GITHUB_TOKEN env var)
        
    Returns:
        True if authentication setup successful, False otherwise
    """
    token = token or os.environ.get('GITHUB_TOKEN')
    
    if token:
        # Update remote URL to include token
        try:
            returncode, stdout, _ = run_git_command(['remote', 'get-url', 'origin'], cwd=repo_path, check=False)
            if returncode == 0:
                url = stdout.strip()
                # Extract repo path from URL
                if 'github.com' in url:
                    # Remove existing token if present
                    if '@' in url:
                        url = url.split('@', 1)[1]
                    if url.startswith('https://'):
                        url = url.replace('https://', '')
                    elif url.startswith('http://'):
                        url = url.replace('http://', '')
                    
                    # Construct new URL with token
                    new_url = f"https://{token}@github.com/{url}"
                    run_git_command(['remote', 'set-url', 'origin', new_url], cwd=repo_path)
                    return True
        except Exception as e:
            print(f"Warning: Could not setup HTTPS token auth: {e}")
            return False
    
    # Check for SSH key
    ssh_key_path = Path.home() / '.ssh' / 'id_rsa'
    if ssh_key_path.exists():
        # SSH should work if key is properly configured
        return True
    
    return False


def add_and_commit(repo_path: Path, message: str, files: Optional[List[str]] = None) -> bool:
    """
    Add files and create a commit.
    
    Args:
        repo_path: Path to repository
        message: Commit message
        files: List of files to add (None = add all changes)
        
    Returns:
        True if commit successful, False otherwise
    """
    try:
        # Setup git config if not already set
        setup_git_config()
        
        # Add files
        if files:
            for file in files:
                file_path = repo_path / file
                if file_path.exists():
                    run_git_command(['add', file], cwd=repo_path)
        else:
            run_git_command(['add', '-A'], cwd=repo_path)
        
        # Check if there are changes to commit
        returncode, stdout, _ = run_git_command(['status', '--porcelain'], cwd=repo_path, check=False)
        if not stdout.strip():
            print("No changes to commit")
            return False
        
        # Create commit
        run_git_command(['commit', '-m', message], cwd=repo_path)
        return True
    except Exception as e:
        print(f"Error creating commit: {e}")
        return False


def push_to_github(repo_path: Path, branch: str = 'main', force: bool = False) -> bool:
    """
    Push changes to GitHub.
    
    Args:
        repo_path: Path to repository
        branch: Branch name (default: main)
        force: Force push (default: False)
        
    Returns:
        True if push successful, False otherwise
    """
    try:
        # Configure Git for large file pushes
        configure_git_for_large_files(repo_path)
        
        # Check authentication
        if not check_git_auth(repo_path):
            print("Warning: Git authentication not configured. Attempting to setup...")
            if not setup_github_auth(repo_path):
                print("Error: Could not setup Git authentication")
                print("Please set GITHUB_TOKEN environment variable or configure SSH keys")
                return False
        
        # Push with increased timeout
        cmd = ['push']
        if force:
            cmd.append('--force')
        cmd.extend(['origin', branch])
        
        # Use environment variables for timeout
        import os
        env = os.environ.copy()
        env['GIT_HTTP_TIMEOUT'] = '300'  # 5 minutes
        
        result = subprocess.run(
            ['git'] + cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            env=env,
            timeout=600  # 10 minute overall timeout
        )
        
        if result.returncode == 0:
            print(f"✓ Successfully pushed to {branch}")
            return True
        else:
            print(f"Error pushing to GitHub: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("Error: Push timed out. Try pushing in smaller chunks or use Git LFS for large files.")
        return False
    except Exception as e:
        print(f"Error pushing to GitHub: {e}")
        return False


def push_models_to_github(
    repo_path: Path,
    model_paths: List[Path],
    commit_message: Optional[str] = None,
    branch: str = 'main',
    skip_large_files: bool = False,
    max_file_size_mb: int = 100
) -> bool:
    """
    Push trained models to GitHub.
    
    Args:
        repo_path: Path to repository
        model_paths: List of model file paths to commit
        commit_message: Custom commit message (default: auto-generated)
        branch: Branch name (default: main)
        skip_large_files: Skip files larger than max_file_size_mb (default: False - push all files)
        max_file_size_mb: Maximum file size in MB to commit (only used if skip_large_files=True)
        
    Returns:
        True if push successful, False otherwise
    """
    if not is_git_repo(repo_path):
        print(f"Error: {repo_path} is not a git repository")
        return False
    
    # Filter model files
    files_to_add = []
    skipped_files = []
    
    for model_path in model_paths:
        if not model_path.exists():
            continue
        
        # Check file size (only if skip_large_files is enabled)
        if skip_large_files:
            size_mb = model_path.stat().st_size / (1024 * 1024)
            if size_mb > max_file_size_mb:
                skipped_files.append((model_path, size_mb))
                print(f"⚠ Skipping large file: {model_path.name} ({size_mb:.2f} MB)")
                continue
        else:
            # Log file size for information
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  Adding: {model_path.name} ({size_mb:.2f} MB)")
        
        # Get relative path from repo root
        try:
            rel_path = model_path.relative_to(repo_path)
            files_to_add.append(str(rel_path))
        except ValueError:
            print(f"Warning: {model_path} is not within repository")
    
    if skipped_files:
        print(f"\n⚠ Skipped {len(skipped_files)} large file(s). Consider using Git LFS for large files.")
    
    if not files_to_add:
        print("No model files to commit")
        return False
    
    # Generate commit message
    if not commit_message:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Add trained models - {timestamp}"
    
    # Add and commit
    print(f"\nCommitting {len(files_to_add)} model file(s)...")
    if not add_and_commit(repo_path, commit_message, files_to_add):
        return False
    
    # Push
    print(f"Pushing to GitHub ({branch})...")
    return push_to_github(repo_path, branch)


if __name__ == '__main__':
    # Test GitHub integration
    import argparse
    
    parser = argparse.ArgumentParser(description='GitHub integration for RunPod')
    parser.add_argument('--repo-path', type=str, default='.', help='Repository path')
    parser.add_argument('--test-auth', action='store_true', help='Test authentication')
    parser.add_argument('--push-models', action='store_true', help='Push model files')
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path).resolve()
    
    if args.test_auth:
        print("Testing Git authentication...")
        if check_git_auth(repo_path):
            print("✓ Authentication configured")
        else:
            print("✗ Authentication not configured")
            print("Set GITHUB_TOKEN environment variable or configure SSH keys")
    
    if args.push_models:
        # Find model files
        model_paths = []
        for pattern in ['models/*.keras', 'PPO approach/models/*.zip', 'PPO approach/checkpoints/**/*.zip']:
            model_paths.extend(repo_path.glob(pattern))
        
        if model_paths:
            push_models_to_github(repo_path, model_paths)
        else:
            print("No model files found")

