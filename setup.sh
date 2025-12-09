#!/bin/bash

echo "=========================================="
echo "Document QA API - Docker Setup"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✓ Docker and Docker Compose found"

# Build and start services
echo ""
echo "Building and starting services..."
docker-compose up -d --build

echo ""
echo "Waiting for services to be healthy..."
sleep 10

# Pull Ollama model
echo ""
echo "Downloading Mistral model (this may take a few minutes)..."
docker exec ollama ollama pull mistral

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Services running:"
echo "  - FastAPI:  http://localhost:8000"
echo "  - Swagger:  http://localhost:8000/docs"
echo "  - Weaviate: http://localhost:8080"
echo "  - Ollama:   http://localhost:11434"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
echo "To restart services:"
echo "  docker-compose restart"
echo ""