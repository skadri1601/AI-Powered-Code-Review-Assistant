# examples/demo.py

"""
Demo script showing how to use the AI Code Review Assistant API.
"""

import asyncio
import httpx
import json
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

# Example diff hunks for demonstration
EXAMPLE_DIFFS = [
    {
        "name": "Add error handling",
        "diff": """@@ -1,5 +1,8 @@
 def process_file(filename):
     with open(filename, 'r') as f:
         data = f.read()
+    if not data:
+        raise ValueError("File is empty")
     return data.upper()"""
    },
    {
        "name": "Function without type hints",
        "diff": """@@ -1,3 +1,3 @@
-def calculate_sum(a, b):
+def calculate_sum(a: int, b: int) -> int:
     return a + b"""
    },
    {
        "name": "Missing docstring",
        "diff": """@@ -1,4 +1,7 @@
 def complex_calculation(x, y, z):
+    \"\"\"Perform complex calculation on three numbers.\"\"\"
     result = x * y + z
     return result if result > 0 else 0"""
    },
    {
        "name": "Potential security issue",
        "diff": """@@ -1,3 +1,4 @@
 import subprocess
 
 def run_command(cmd):
+    # TODO: Validate command input
     return subprocess.run(cmd, shell=True, capture_output=True)"""
    }
]

async def test_api_health():
    """Test API health endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                health_data = response.json()
                print("✅ API Health Check:")
                print(f"   Status: {health_data['status']}")
                print(f"   Model: {health_data['model_status']}")
                print(f"   GitHub App: {health_data.get('github_app_configured', 'Not configured')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to API: {e}")
            return False

async def test_review_generation():
    """Test review generation with example diffs."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("\n🧠 Testing AI Review Generation:")
        print("-" * 50)
        
        for i, example in enumerate(EXAMPLE_DIFFS, 1):
            print(f"\n{i}. {example['name']}:")
            print("   Diff:")
            for line in example['diff'].split('\n'):
                if line.startswith('+'):
                    print(f"   \033[92m{line}\033[0m")  # Green for additions
                elif line.startswith('-'):
                    print(f"   \033[91m{line}\033[0m")  # Red for deletions
                else:
                    print(f"   {line}")
            
            try:
                response = await client.post(
                    f"{API_BASE_URL}/review",
                    json={
                        "diff_hunks": [example['diff']],
                        "max_length": 64,
                        "num_beams": 4
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    suggestion = data['suggestions'][0]
                    print(f"   \033[94mAI Suggestion: {suggestion}\033[0m")  # Blue for suggestions
                else:
                    print(f"   ❌ Error: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Request failed: {e}")

async def test_multiple_hunks():
    """Test review generation with multiple diff hunks."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("\n📦 Testing Multiple Hunks:")
        print("-" * 30)
        
        all_diffs = [example['diff'] for example in EXAMPLE_DIFFS]
        
        try:
            response = await client.post(
                f"{API_BASE_URL}/review",
                json={
                    "diff_hunks": all_diffs,
                    "max_length": 96,
                    "num_beams": 3
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                suggestions = data['suggestions']
                model_info = data['model_info']
                
                print(f"✅ Processed {model_info['num_hunks_processed']} hunks")
                print(f"   Model: {model_info['model_path']}")
                print(f"   Parameters: {model_info['parameters']}")
                
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"\n   Suggestion {i}: {suggestion}")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")

async def test_performance():
    """Test API performance with concurrent requests."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("\n⚡ Performance Test:")
        print("-" * 20)
        
        test_diff = EXAMPLE_DIFFS[0]['diff']
        
        async def make_request():
            response = await client.post(
                f"{API_BASE_URL}/review",
                json={"diff_hunks": [test_diff]}
            )
            return response.status_code == 200
        
        # Test concurrent requests
        import time
        start_time = time.time()
        
        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        successful = sum(1 for r in results if r is True)
        total_time = end_time - start_time
        
        print(f"   Concurrent requests: 5")
        print(f"   Successful: {successful}/5")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average per request: {total_time/5:.2f}s")

def save_demo_results():
    """Save demo results for documentation."""
    demo_data = {
        "examples": EXAMPLE_DIFFS,
        "api_endpoints": {
            "health": f"{API_BASE_URL}/health",
            "review": f"{API_BASE_URL}/review",
            "docs": f"{API_BASE_URL}/docs"
        },
        "usage_examples": {
            "curl": f"""curl -X POST "{API_BASE_URL}/review" \\
  -H "Content-Type: application/json" \\
  -d '{{"diff_hunks": ["@@ -1,1 +1,2 @@\\n def test():\\n+    pass"]}}'""",
            "python": """
import httpx

async def get_review():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/review",
            json={"diff_hunks": ["@@ -1,1 +1,2 @@\\n def test():\\n+    pass"]}
        )
        return response.json()
"""
        }
    }
    
    output_file = Path("examples/demo_results.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(demo_data, f, indent=2)
    
    print(f"\n💾 Demo results saved to {output_file}")

async def main():
    """Run the complete demo."""
    print("🤖 AI Code Review Assistant - Demo")
    print("=" * 50)
    
    # Check API health
    if not await test_api_health():
        print("\n❌ API is not healthy. Please start the service first:")
        print("   make run-dev")
        return
    
    # Test individual examples
    await test_review_generation()
    
    # Test multiple hunks
    await test_multiple_hunks()
    
    # Performance test
    await test_performance()
    
    # Save results
    save_demo_results()
    
    print("\n🎉 Demo completed successfully!")
    print(f"   API Documentation: {API_BASE_URL}/docs")
    print(f"   Interactive Testing: {API_BASE_URL}/docs#/default/review_review_post")

if __name__ == "__main__":
    asyncio.run(main())