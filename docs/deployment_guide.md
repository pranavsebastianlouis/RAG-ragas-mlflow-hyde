# Deployment Guide

## 📋 Prerequisites

### System Requirements

- **Docker Desktop** 20.10+ ([Download](https://www.docker.com/products/docker-desktop))
- **Docker Compose** 1.29+ (included with Docker Desktop)
- **8GB RAM** minimum (allocated to Docker)
- **10GB free disk space**

### For Local Development

- **Python** 3.11+
- **pip** package manager
- **Ollama** (for local LLM)

## 🚀 Quick Start with Docker (Recommended)

### Step 1: Clone and Setup

```bash
# Clone repository
git clone <your-repo>
cd document-qa

# Verify files
ls -la
# Should see: doc_qa_chatbot.py, docker-compose.yml, requirements.txt, etc.
```

### Step 2: Build and Start Services

```bash
# Build and start all services
docker-compose up -d --build

# Expected output:
# Creating network "app-network"...
# Creating volume "weaviate_data"...
# Creating volume "ollama_data"...
# Creating weaviate...done
# Creating ollama...done
# Creating fastapi_app...done
```

### Step 3: Download LLM Model

```bash
# Download Mistral 7B model
docker exec ollama ollama pull mistral

# Verify model is downloaded
docker exec ollama ollama list

# Expected output:
# NAME            ID              SIZE
# mistral:latest  xxx             4.1GB
```

### Step 4: Verify Services

```bash
# Check all services are running
docker-compose ps

# Should show:
# NAME          STATUS       PORTS
# weaviate      Up          0.0.0.0:8080->8080/tcp
# ollama        Up          0.0.0.0:11434->11434/tcp
# fastapi_app   Up          0.0.0.0:8000->8000/tcp

# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "weaviate_connected": true,
#   "llm_ready": true,
#   "embeddings_ready": true
# }
```

### Step 5: Access Services

- **FastAPI:** http://localhost:8000
- **Swagger UI:** http://localhost:8000/docs
- **Weaviate:** http://localhost:8080
- **Ollama:** http://localhost:11434

## 💻 Local Development Setup

### Step 1: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Start Weaviate

```bash
# Run Weaviate in Docker
docker run -d \
  -p 8080:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e DEFAULT_VECTORIZER_MODULE=none \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -v weaviate_data:/var/lib/weaviate \
  --name weaviate \
  semitechnologies/weaviate:latest

# Verify Weaviate is ready
curl http://localhost:8080/v1/.well-known/ready
```

### Step 3: Start Ollama

```bash
# Start Ollama server
ollama serve &

# Download Mistral model
ollama pull mistral

# Verify
ollama list
```

### Step 4: Run Application

```bash
# Start FastAPI application
python doc_qa_chatbot.py

# Application will start on http://localhost:8000
# Press Ctrl+C to stop
```

## 📦 Configuration

### Environment Variables

Create a `.env` file (optional):

```bash
# Weaviate Configuration
WEAVIATE_URL=http://localhost:8080

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Model Configuration
OLLAMA_MODEL=mistral

# Chunking Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=100

# Retrieval Configuration
TOP_K=5
```

### Docker Compose Configuration

Edit `docker-compose.yml` to customize:

```yaml
version: '3.8'

services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      DEFAULT_VECTORIZER_MODULE: 'none'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
    volumes:
      - weaviate_data:/var/lib/weaviate
    networks:
      - app-network

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    networks:
      - app-network

  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - WEAVIATE_URL=http://weaviate:8080
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - weaviate
      - ollama
    volumes:
      - ./mlruns:/app/mlruns
      - ./uploads:/app/uploads
    networks:
      - app-network

volumes:
  weaviate_data:
  ollama_data:

networks:
  app-network:
    driver: bridge
```

## 🐳 Docker Commands Reference

### Service Management

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart services
docker-compose restart

# Stop and remove volumes (clean slate)
docker-compose down -v

# Rebuild and restart
docker-compose up -d --build
```

### Viewing Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f app
docker-compose logs -f weaviate
docker-compose logs -f ollama

# Last 100 lines
docker-compose logs --tail=100 app
```

### Container Access

```bash
# Access FastAPI container
docker exec -it fastapi_app bash

# Access Ollama container
docker exec -it ollama bash

# Access Weaviate container
docker exec -it weaviate bash
```

### Health Checks

```bash
# Check service status
docker-compose ps

# Check resource usage
docker stats

# Inspect container
docker inspect fastapi_app
```

## 🔧 Troubleshooting

### Service Won't Start

```bash
# Check logs for errors
docker-compose logs app

# Verify port availability
netstat -ano | findstr :8000  # Windows
lsof -i :8000                  # Mac/Linux

# Restart service
docker-compose restart app
```

### Ollama Model Not Found

```bash
# Re-pull the model
docker exec ollama ollama pull mistral

# List available models
docker exec ollama ollama list

# Check Ollama logs
docker-compose logs ollama
```

### Weaviate Connection Issues

```bash
# Check Weaviate health
curl http://localhost:8080/v1/.well-known/ready

# Should return: {"status":"ok"}

# Check Weaviate logs
docker-compose logs weaviate

# Restart Weaviate
docker-compose restart weaviate
```

### Out of Memory Errors

```bash
# Check Docker memory settings
docker info | grep Memory

# Increase in Docker Desktop:
# Settings → Resources → Memory → 8GB+

# Check container memory usage
docker stats
```

### Port Already in Use

```bash
# Find process using port
netstat -ano | findstr :8000  # Windows
lsof -i :8000                  # Mac/Linux

# Kill process or change port in docker-compose.yml
ports:
  - "8001:8000"  # Use different host port
```

### Application Errors

```bash
# Check FastAPI logs
docker-compose logs -f app

# Check health endpoint
curl http://localhost:8000/health

# Check debug status
curl http://localhost:8000/debug/status
```

## 🗄️ Data Persistence

### Backing Up Data

```bash
# Backup Weaviate data
docker run --rm \
  -v doc_qa_weaviate_data:/data \
  -v $(pwd):/backup \
  busybox tar czf /backup/weaviate_backup.tar.gz /data

# Backup MLflow data
tar czf mlflow_backup.tar.gz mlruns/

# Backup Ollama models
docker run --rm \
  -v doc_qa_ollama_data:/data \
  -v $(pwd):/backup \
  busybox tar czf /backup/ollama_backup.tar.gz /data
```

### Restoring Data

```bash
# Restore Weaviate data
docker run --rm \
  -v doc_qa_weaviate_data:/data \
  -v $(pwd):/backup \
  busybox tar xzf /backup/weaviate_backup.tar.gz -C /

# Restore MLflow data
tar xzf mlflow_backup.tar.gz
```

### Volume Management

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect doc_qa_weaviate_data

# Remove unused volumes
docker volume prune

# Remove specific volume (WARNING: deletes data)
docker volume rm doc_qa_weaviate_data
```

## 🧹 Cleanup

### Remove Containers and Networks

```bash
# Stop and remove containers
docker-compose down

# Remove containers, networks, and images
docker-compose down --rmi all

# Remove everything including volumes (DELETE ALL DATA)
docker-compose down -v --rmi all
```

### Clean Docker System

```bash
# Remove unused containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove everything unused
docker system prune -a --volumes
```

## 🔐 Production Deployment

### Security Hardening

```bash
# 1. Enable authentication
# Edit docker-compose.yml for Weaviate:
environment:
  AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'false'
  AUTHENTICATION_APIKEY_ENABLED: 'true'
  AUTHENTICATION_APIKEY_ALLOWED_KEYS: 'your-secret-key'

# 2. Use secrets for sensitive data
# Create docker-compose.override.yml:
services:
  app:
    environment:
      - WEAVIATE_API_KEY=/run/secrets/weaviate_key
    secrets:
      - weaviate_key

secrets:
  weaviate_key:
    file: ./secrets/weaviate_key.txt
```

### Reverse Proxy (Nginx)

```nginx
# nginx.conf
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### SSL/TLS Configuration

```bash
# Using Let's Encrypt with Certbot
sudo certbot --nginx -d your-domain.com

# Or add to nginx config:
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
    }
}
```

### Resource Limits

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Health Checks

```yaml
# docker-compose.yml
services:
  app:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## 📊 Monitoring

### View MLflow UI

```bash
# Start MLflow UI
mlflow ui --port 5000 --host 0.0.0.0

# Access at: http://localhost:5000
```

### Application Logs

```bash
# Real-time logs
docker-compose logs -f app

# Last 100 lines
docker-compose logs --tail=100 app

# Save logs to file
docker-compose logs app > app_logs.txt
```

### Resource Monitoring

```bash
# Monitor container resources
docker stats

# Monitor specific container
docker stats fastapi_app

# Export metrics
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

## 🚀 Scaling

### Horizontal Scaling (Multiple Instances)

```yaml
# docker-compose.yml
services:
  app:
    build: .
    deploy:
      replicas: 3
    ports:
      - "8000-8002:8000"
```

### Load Balancing with Nginx

```nginx
upstream api_backend {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    location / {
        proxy_pass http://api_backend;
    }
}
```

## ✅ Deployment Checklist

- [ ] Docker and Docker Compose installed
- [ ] 8GB+ RAM allocated to Docker
- [ ] All services start without errors
- [ ] Mistral model downloaded
- [ ] Health check returns "healthy"
- [ ] Can upload PDF successfully
- [ ] Can query and get answers
- [ ] MLflow tracking works
- [ ] Evaluation endpoints functional
- [ ] Data persistence configured
- [ ] Backups scheduled (production)
- [ ] Monitoring configured (production)
- [ ] Security hardening applied (production)

## 📞 Support

If you encounter issues:

1. **Check logs:** `docker-compose logs -f`
2. **Verify health:** `curl http://localhost:8000/health`
3. **Check resources:** `docker stats`
4. **Review documentation:** See other docs in `/docs`

---

