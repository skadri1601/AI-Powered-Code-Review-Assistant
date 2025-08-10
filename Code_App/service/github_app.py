# service/github_app.py

"""
GitHub App integration for automated code review.
Handles webhook events and posts review comments.
"""

import os
import jwt
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import httpx
from fastapi import HTTPException
from github import Github, GithubIntegration
from cryptography.hazmat.primitives import serialization

class GitHubAppClient:
    def __init__(self):
        self.app_id = os.getenv("GITHUB_APP_ID")
        self.private_key_path = os.getenv("GITHUB_PRIVATE_KEY_PATH", "github-app-key.pem")
        self.webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET")
        
        if not all([self.app_id, self.webhook_secret]):
            raise ValueError("Missing required GitHub App configuration")
        
        # Load private key
        try:
            with open(self.private_key_path, 'rb') as key_file:
                self.private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None
                )
        except FileNotFoundError:
            raise ValueError(f"Private key file not found: {self.private_key_path}")
    
    def get_jwt_token(self) -> str:
        """Generate JWT token for GitHub App authentication."""
        now = int(time.time())
        payload = {
            'iat': now - 60,  # Issued at (60 seconds ago to account for clock skew)
            'exp': now + 600,  # Expires in 10 minutes
            'iss': self.app_id
        }
        
        return jwt.encode(payload, self.private_key, algorithm='RS256')
    
    def get_installation_token(self, installation_id: int) -> str:
        """Get installation access token."""
        jwt_token = self.get_jwt_token()
        
        headers = {
            'Authorization': f'Bearer {jwt_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        
        return response.json()['token']
    
    def get_github_client(self, installation_id: int) -> Github:
        """Get authenticated GitHub client for installation."""
        token = self.get_installation_token(installation_id)
        return Github(token)
    
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook payload signature."""
        import hmac
        import hashlib
        
        expected = hmac.new(
            self.webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        received = signature.replace('sha256=', '')
        return hmac.compare_digest(expected, received)

class CodeReviewBot:
    def __init__(self, github_client: GitHubAppClient, review_service_url: str):
        self.github_client = github_client
        self.review_service_url = review_service_url
    
    async def handle_pull_request(self, event_data: Dict[str, Any]) -> None:
        """Handle pull request events."""
        action = event_data.get('action')
        
        # Only process opened and synchronized (updated) PRs
        if action not in ['opened', 'synchronize']:
            return
        
        pr_data = event_data['pull_request']
        repo_data = event_data['repository']
        installation_id = event_data['installation']['id']
        
        try:
            # Get GitHub client
            gh = self.github_client.get_github_client(installation_id)
            repo = gh.get_repo(repo_data['full_name'])
            pr = repo.get_pull(pr_data['number'])
            
            # Get PR diff
            diff_hunks = await self._extract_diff_hunks(pr)
            
            if not diff_hunks:
                return
            
            # Get AI review suggestions
            suggestions = await self._get_review_suggestions(diff_hunks)
            
            # Post review comments
            await self._post_review_comments(pr, diff_hunks, suggestions)
            
        except Exception as e:
            print(f"Error processing PR {pr_data['number']}: {e}")
            # Optionally post a comment about the error
            # await self._post_error_comment(pr, str(e))
    
    async def _extract_diff_hunks(self, pr) -> list[str]:
        """Extract diff hunks from PR."""
        files = pr.get_files()
        hunks = []
        
        for file in files:
            if file.patch:  # Only files with changes
                # Filter for code files (avoid binaries, docs, etc.)
                if self._is_code_file(file.filename):
                    # Split patch into hunks
                    patch_hunks = self._split_patch_into_hunks(file.patch)
                    hunks.extend(patch_hunks)
        
        return hunks[:10]  # Limit to 10 hunks to avoid API limits
    
    def _is_code_file(self, filename: str) -> bool:
        """Check if file is a code file worth reviewing."""
        code_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.cs', '.php', '.rb', '.go', '.rs', '.kt', '.swift',
            '.scala', '.clj', '.hs', '.ml', '.r', '.sql', '.sh'
        }
        
        # Exclude certain directories/files
        excluded_patterns = {
            'node_modules/', 'vendor/', '.git/', '__pycache__/',
            '.pytest_cache/', 'dist/', 'build/', 'target/',
            'package-lock.json', 'yarn.lock', 'Cargo.lock'
        }
        
        for pattern in excluded_patterns:
            if pattern in filename:
                return False
        
        return any(filename.endswith(ext) for ext in code_extensions)
    
    def _split_patch_into_hunks(self, patch: str) -> list[str]:
        """Split a git patch into individual hunks."""
        hunks = []
        current_hunk = []
        
        for line in patch.split('\n'):
            if line.startswith('@@'):
                if current_hunk:
                    hunks.append('\n'.join(current_hunk))
                current_hunk = [line]
            elif current_hunk:
                current_hunk.append(line)
        
        if current_hunk:
            hunks.append('\n'.join(current_hunk))
        
        return hunks
    
    async def _get_review_suggestions(self, diff_hunks: list[str]) -> list[str]:
        """Get AI review suggestions for diff hunks."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.review_service_url}/review",
                json={"diff_hunks": diff_hunks},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()['suggestions']
    
    async def _post_review_comments(self, pr, diff_hunks: list[str], suggestions: list[str]) -> None:
        """Post review comments on PR."""
        if len(suggestions) != len(diff_hunks):
            print(f"Mismatch: {len(suggestions)} suggestions for {len(diff_hunks)} hunks")
            return
        
        meaningful_suggestions = []
        for suggestion in suggestions:
            # Filter out generic/unhelpful suggestions
            if self._is_meaningful_suggestion(suggestion):
                meaningful_suggestions.append(suggestion)
        
        if not meaningful_suggestions:
            return
        
        # Create a single review with all suggestions
        review_body = "## 🤖 AI Code Review\n\n"
        review_body += "Here are some suggestions from the AI code review assistant:\n\n"
        
        for i, (hunk, suggestion) in enumerate(zip(diff_hunks, suggestions)):
            if self._is_meaningful_suggestion(suggestion):
                review_body += f"### Suggestion {i+1}\n"
                review_body += f"```diff\n{hunk[:200]}{'...' if len(hunk) > 200 else ''}\n```\n"
                review_body += f"**Suggestion:** {suggestion}\n\n"
        
        review_body += "\n*This review was generated by an AI assistant. Please use your judgment when applying suggestions.*"
        
        # Post as a PR review
        pr.create_review(body=review_body, event="COMMENT")
    
    def _is_meaningful_suggestion(self, suggestion: str) -> bool:
        """Filter out generic or unhelpful suggestions."""
        # Skip very short suggestions
        if len(suggestion.strip()) < 10:
            return False
        
        # Skip generic phrases
        generic_phrases = [
            "looks good", "lgtm", "ok", "fine", "good job",
            "nice work", "well done", "no issues", "approved"
        ]
        
        suggestion_lower = suggestion.lower()
        return not any(phrase in suggestion_lower for phrase in generic_phrases)