# tests/test_app.py

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

# Mock the transformers imports to avoid loading actual models in tests
with patch('transformers.AutoTokenizer'), \
     patch('transformers.AutoModelForSeq2SeqLM'), \
     patch('transformers.pipeline'):
    from service.app import app

client = TestClient(app)

class TestAPIEndpoints:
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_status" in data

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "endpoints" in data

    @patch('service.app.get_pipeline')
    def test_review_endpoint_success(self, mock_pipeline):
        """Test successful review generation."""
        # Mock the pipeline
        mock_gen = Mock()
        mock_gen.return_value = [{"generated_text": "Consider adding error handling here."}]
        mock_pipeline.return_value = mock_gen
        
        request_data = {
            "diff_hunks": [
                "@@ -1,3 +1,4 @@\n def test():\n+    x = 1\n     pass"
            ]
        }
        
        response = client.post("/review", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert "model_info" in data
        assert len(data["suggestions"]) == 1

    @patch('service.app.get_pipeline')
    def test_review_endpoint_multiple_hunks(self, mock_pipeline):
        """Test review with multiple diff hunks."""
        mock_gen = Mock()
        mock_gen.side_effect = [
            [{"generated_text": "Suggestion 1"}],
            [{"generated_text": "Suggestion 2"}]
        ]
        mock_pipeline.return_value = mock_gen
        
        request_data = {
            "diff_hunks": ["hunk1", "hunk2"],
            "max_length": 128,
            "num_beams": 2
        }
        
        response = client.post("/review", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert len(data["suggestions"]) == 2
        assert data["model_info"]["parameters"]["max_length"] == 128

    def test_review_endpoint_empty_hunks(self):
        """Test review with empty hunks list."""
        request_data = {"diff_hunks": []}
        response = client.post("/review", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["suggestions"] == []

    @patch('service.app.github_client', None)
    def test_webhook_endpoint_not_configured(self):
        """Test webhook when GitHub App not configured."""
        response = client.post("/webhook", json={})
        assert response.status_code == 501

    def test_webhook_endpoint_missing_signature(self):
        """Test webhook with missing signature."""
        with patch('service.app.review_bot', Mock()):
            response = client.post("/webhook", json={})
            assert response.status_code == 400


class TestDataProcessing:
    
    def test_extract_diff_hunks(self):
        """Test diff hunk extraction."""
        from scripts.download_pr_data import extract_diff_hunks
        
        diff_text = """@@ -1,3 +1,4 @@
 def hello():
+    print("Hello")
     pass
@@ -10,2 +11,3 @@
 def world():
+    print("World")
     pass"""
        
        hunks = extract_diff_hunks(diff_text)
        assert len(hunks) == 2
        assert "@@ -1,3 +1,4 @@" in hunks[0]
        assert "@@ -10,2 +11,3 @@" in hunks[1]

    def test_load_pairs(self):
        """Test loading training pairs."""
        import tempfile
        import json
        from scripts.preprocess import load_pairs
        
        # Create temporary JSONL file
        test_data = [
            {"diff_hunk": "test_hunk_1", "review_comment": "test_comment_1"},
            {"diff_hunk": "test_hunk_2", "review_comment": "test_comment_2"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        pairs = list(load_pairs(temp_path))
        assert len(pairs) == 2
        assert pairs[0]["input"] == "test_hunk_1"
        assert pairs[0]["target"] == "test_comment_1"
        
        # Cleanup
        import os
        os.unlink(temp_path)


class TestGitHubIntegration:
    
    @patch('service.github_app.requests.post')
    def test_get_installation_token(self, mock_post):
        """Test GitHub installation token retrieval."""
        from service.github_app import GitHubAppClient
        
        # Mock environment variables
        with patch.dict('os.environ', {
            'GITHUB_APP_ID': '12345',
            'GITHUB_WEBHOOK_SECRET': 'secret'
        }):
            # Mock private key file
            with patch('builtins.open', mock_open_key()):
                with patch('service.github_app.serialization.load_pem_private_key') as mock_key:
                    mock_key.return_value = Mock()
                    
                    client = GitHubAppClient()
                    
                    # Mock API response
                    mock_response = Mock()
                    mock_response.json.return_value = {'token': 'test_token'}
                    mock_response.raise_for_status.return_value = None
                    mock_post.return_value = mock_response
                    
                    token = client.get_installation_token(123)
                    assert token == 'test_token'

    def test_verify_webhook_signature(self):
        """Test webhook signature verification."""
        from service.github_app import GitHubAppClient
        
        with patch.dict('os.environ', {
            'GITHUB_APP_ID': '12345',
            'GITHUB_WEBHOOK_SECRET': 'secret'
        }):
            with patch('builtins.open', mock_open_key()):
                with patch('service.github_app.serialization.load_pem_private_key'):
                    client = GitHubAppClient()
                    
                    # Test valid signature
                    import hmac
                    import hashlib
                    payload = b'{"test": "data"}'
                    expected = hmac.new(b'secret', payload, hashlib.sha256).hexdigest()
                    signature = f'sha256={expected}'
                    
                    assert client.verify_webhook_signature(payload, signature)
                    
                    # Test invalid signature
                    assert not client.verify_webhook_signature(payload, 'sha256=invalid')

    def test_is_code_file(self):
        """Test code file detection."""
        from service.github_app import CodeReviewBot
        
        bot = CodeReviewBot(Mock(), "http://test")
        
        # Test code files
        assert bot._is_code_file("main.py")
        assert bot._is_code_file("app.js")
        assert bot._is_code_file("Component.tsx")
        assert bot._is_code_file("service.go")
        
        # Test non-code files
        assert not bot._is_code_file("README.md")
        assert not bot._is_code_file("package-lock.json")
        assert not bot._is_code_file("node_modules/package.json")
        assert not bot._is_code_file("dist/bundle.js")

    def test_is_meaningful_suggestion(self):
        """Test meaningful suggestion filtering."""
        from service.github_app import CodeReviewBot
        
        bot = CodeReviewBot(Mock(), "http://test")
        
        # Meaningful suggestions
        assert bot._is_meaningful_suggestion("Consider adding error handling for this network request.")
        assert bot._is_meaningful_suggestion("This function could benefit from type hints.")
        assert bot._is_meaningful_suggestion("Memory leak potential - consider using context manager.")
        
        # Non-meaningful suggestions
        assert not bot._is_meaningful_suggestion("lgtm")
        assert not bot._is_meaningful_suggestion("looks good")
        assert not bot._is_meaningful_suggestion("ok")
        assert not bot._is_meaningful_suggestion("short")


def mock_open_key():
    """Mock file open for private key."""
    from unittest.mock import mock_open
    key_content = b"""-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA4f5wg5l2hKsTeNem/V41fGnJm6gOdrj8ym3rFkEjWT...
-----END RSA PRIVATE KEY-----"""
    return mock_open(read_data=key_content)


# Integration Tests
class TestEndToEndWorkflow:
    
    @pytest.mark.asyncio
    @patch('service.app.get_pipeline')
    async def test_full_review_workflow(self, mock_pipeline):
        """Test complete review workflow."""
        # Mock pipeline
        mock_gen = Mock()
        mock_gen.return_value = [{"generated_text": "Add input validation here."}]
        mock_pipeline.return_value = mock_gen
        
        # Test data
        diff_hunk = """@@ -1,5 +1,8 @@
 def process_user_input(data):
+    if not data:
+        raise ValueError("Data cannot be empty")
     result = data.upper()
     return result"""
        
        request_data = {"diff_hunks": [diff_hunk]}
        
        response = client.post("/review", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["suggestions"]) == 1
        assert "model_info" in data
        assert data["model_info"]["num_hunks_processed"] == 1


# Performance Tests
class TestPerformance:
    
    @patch('service.app.get_pipeline')
    def test_large_diff_handling(self, mock_pipeline):
        """Test handling of large diff hunks."""
        mock_gen = Mock()
        mock_gen.return_value = [{"generated_text": "Suggestion for large diff."}]
        mock_pipeline.return_value = mock_gen
        
        # Create a very large diff hunk
        large_hunk = "@@ -1,100 +1,200 @@\n" + "\n".join([f"+ line {i}" for i in range(1000)])
        
        request_data = {"diff_hunks": [large_hunk]}
        
        response = client.post("/review", json=request_data)
        assert response.status_code == 200
        
        # Verify the hunk was truncated (our app truncates at 512 chars)
        mock_gen.assert_called_once()
        called_hunk = mock_gen.call_args[0][0]
        assert len(called_hunk) <= 515  # 512 + "..."

    @patch('service.app.get_pipeline')
    def test_multiple_hunks_performance(self, mock_pipeline):
        """Test performance with multiple hunks."""
        mock_gen = Mock()
        mock_gen.return_value = [{"generated_text": "Test suggestion."}]
        mock_pipeline.return_value = mock_gen
        
        # Create multiple small hunks
        hunks = [f"@@ -1,1 +1,2 @@\n function{i}()" for i in range(10)]
        
        request_data = {"diff_hunks": hunks}
        
        import time
        start_time = time.time()
        response = client.post("/review", json=request_data)
        end_time = time.time()
        
        assert response.status_code == 200
        assert len(response.json()["suggestions"]) == 10
        # Should complete within reasonable time (adjust as needed)
        assert (end_time - start_time) < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])