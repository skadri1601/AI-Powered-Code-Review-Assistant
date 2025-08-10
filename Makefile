# Makefile

.PHONY: help install data train test build run clean deploy

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
COMPOSE := docker-compose

# Help
help:
	@echo "AI Code Review Assistant - Available Commands:"
	@echo ""
	@echo "Setup & Dependencies:"
	@echo "  install          Install Python dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo ""
	@echo "Data & Training:"
	@echo "  data             Download and preprocess training data"
	@echo "  train            Fine-tune the CodeT5 model"
	@echo "  evaluate         Evaluate model performance"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  test             Run test suite"
	@echo "  test-cov         Run tests with coverage report"
	@echo "  lint             Run code linting"
	@echo "  format           Format code with black"
	@echo ""
	@echo "Docker & Deployment:"
	@echo "  build            Build Docker image"
	@echo "  run              Run with Docker Compose"
	@echo "  run-dev          Run in development mode"
	@echo "  stop             Stop all services"
	@echo "  logs             View application logs"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            Clean up generated files"
	@echo "  clean-data       Clean data directory"
	@echo "  clean-models     Clean model checkpoints"

# Setup & Dependencies
install:
	$(PIP) install -r scripts/requirements.txt
	$(PIP) install fastapi uvicorn[standard]

install-dev: install
	$(PIP) install pytest pytest-cov pytest-asyncio black flake8 httpx

# Data & Training Pipeline
data: data/pr_review_pairs.jsonl

data/pr_review_pairs.jsonl:
	@echo "📥 Downloading PR review data..."
	$(PYTHON) scripts/download_pr_data.py
	@echo "🔄 Preprocessing data..."
	$(PYTHON) scripts/preprocess.py
	@echo "✅ Data pipeline complete!"

train: data
	@echo "🏋️ Starting model training..."
	$(PYTHON) scripts/finetune.py
	@echo "✅ Training complete!"

evaluate:
	@echo "📊 Evaluating model performance..."
	$(PYTHON) -c "
import json
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load test data
test_file = Path('data/hf_dataset/test.jsonl')
if not test_file.exists():
    print('❌ Test data not found. Run make data first.')
    exit(1)

# Load model
model_dir = 'checkpoints/codet5-finetuned/final'
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    gen = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=-1)
    print('✅ Model loaded successfully')
except Exception as e:
    print(f'❌ Failed to load model: {e}')
    exit(1)

# Evaluate on sample
with open(test_file) as f:
    test_cases = [json.loads(line) for line in f][:10]

print(f'🧪 Testing on {len(test_cases)} samples...')
for i, case in enumerate(test_cases):
    prediction = gen(case['input'], max_length=64, num_beams=4)[0]['generated_text']
    print(f'Sample {i+1}:')
    print(f'  Input: {case[\"input\"][:100]}...')
    print(f'  Expected: {case[\"target\"]}')
    print(f'  Predicted: {prediction}')
    print()
"

# Testing & Quality
test:
	@echo "🧪 Running test suite..."
	$(PYTHON) -m pytest tests/ -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	$(PYTHON) -m pytest tests/ -v --cov=scripts --cov=service --cov-report=html --cov-report=term

lint:
	@echo "🔍 Linting code..."
	flake8 scripts/ service/ tests/ --max-line-length=100 --ignore=E203,W503
	@echo "✅ Linting complete!"

format:
	@echo "🎨 Formatting code..."
	black scripts/ service/ tests/ --line-length=100
	@echo "✅ Formatting complete!"

# Docker & Deployment
build:
	@echo "🐳 Building Docker image..."
	$(DOCKER) build -t ai-code-review .
	@echo "✅ Docker image built!"

run: build
	@echo "🚀 Starting services with Docker Compose..."
	$(COMPOSE) up -d
	@echo "✅ Services started! API available at http://localhost:8000"

run-dev:
	@echo "🚀 Starting development server..."
	$(PYTHON) -m uvicorn service.app:app --reload --host 0.0.0.0 --port 8000

stop:
	@echo "🛑 Stopping services..."
	$(COMPOSE) down

logs:
	@echo "📋 Viewing application logs..."
	$(COMPOSE) logs -f review-assistant

# GitHub App Setup
setup-github-app:
	@echo "🔧 GitHub App Setup Instructions:"
	@echo ""
	@echo "1. Create a new GitHub App at https://github.com/settings/apps/new"
	@echo "2. Configure the following settings:"
	@echo "   - Name: AI Code Review Assistant"
	@echo "   - Homepage URL: https://your-domain.com"
	@echo "   - Webhook URL: https://your-domain.com/webhook"
	@echo "   - Webhook secret: Generate a random string"
	@echo ""
	@echo "3. Set permissions:"
	@echo "   - Pull requests: Read & Write"
	@echo "   - Contents: Read"
	@echo "   - Metadata: Read"
	@echo ""
	@echo "4. Subscribe to events:"
	@echo "   - Pull request"
	@echo ""
	@echo "5. Download the private key and save as 'secrets/github-app-key.pem'"
	@echo "6. Set environment variables:"
	@echo "   - GITHUB_APP_ID=<your-app-id>"
	@echo "   - GITHUB_WEBHOOK_SECRET=<your-webhook-secret>"
	@echo ""
	@echo "7. Install the app on your repositories"

# Utilities
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	@echo "✅ Cleanup complete!"

clean-data:
	@echo "🧹 Cleaning data directory..."
	rm -rf data/pr_review_pairs.jsonl
	rm -rf data/hf_dataset/
	@echo "✅ Data cleanup complete!"

clean-models:
	@echo "🧹 Cleaning model checkpoints..."
	rm -rf checkpoints/
	@echo "✅ Model cleanup complete!"

# Development workflow
dev-setup: install-dev data
	@echo "🎯 Development environment ready!"
	@echo "💡 Next steps:"
	@echo "   1. Run 'make train' to train the model"
	@echo "   2. Run 'make run-dev' to start the development server"
	@echo "   3. Run 'make test' to run the test suite"

# Production deployment check
deploy-check:
	@echo "🔍 Pre-deployment checks..."
	@echo "Checking required files..."
	@test -f .env || (echo "❌ .env file missing" && exit 1)
	@test -f secrets/github-app-key.pem || (echo "❌ GitHub App private key missing" && exit 1)
	@test -d checkpoints/codet5-finetuned/final || (echo "❌ Trained model missing" && exit 1)
	@echo "Checking environment variables..."
	@test -n "$$GITHUB_APP_ID" || (echo "❌ GITHUB_APP_ID not set" && exit 1)
	@test -n "$$GITHUB_WEBHOOK_SECRET" || (echo "❌ GITHUB_WEBHOOK_SECRET not set" && exit 1)
	@echo "✅ All checks passed!"

# Quick start for new users
quickstart:
	@echo "🚀 AI Code Review Assistant - Quick Start"
	@echo ""
	@echo "This will set up the complete development environment..."
	@make dev-setup
	@echo ""
	@echo "🎉 Setup complete! Your AI Code Review Assistant is ready."
	@echo ""
	@echo "📝 Next steps:"
	@echo "   1. Train the model: make train"
	@echo "   2. Start development server: make run-dev"
	@echo "   3. Test the API at http://localhost:8000/docs"
	@echo "   4. Set up GitHub App: make setup-github-app"