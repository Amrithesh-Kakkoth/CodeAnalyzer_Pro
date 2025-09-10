# 🐳 Docker Setup for Code Quality Intelligence Agent

## Quick Start

### 1. **Build the Docker Image**
```bash
docker build -t code-quality-agent .
```

### 2. **Run with Docker Compose (Recommended)**

**Interactive Chat Mode:**
```bash
docker-compose up code-quality-agent
```

**Analysis Mode:**
```bash
docker-compose up code-quality-analyzer
```

### 3. **Run with Docker Commands**

**Interactive Chat:**
```bash
docker run -it --rm \
  -v $(pwd):/workspace:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.env:/app/.env:ro \
  -w /workspace \
  code-quality-agent \
  python -m code_quality_agent chat . --enhanced
```

**Analysis:**
```bash
docker run --rm \
  -v $(pwd):/workspace:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.env:/app/.env:ro \
  -w /workspace \
  code-quality-agent \
  python -m code_quality_agent analyze .
```

## 📁 **Volume Mounts**

- `/workspace` - Your code directory (read-only)
- `/app/data` - Vector databases (persistent)
- `/.env` - Environment variables (read-only)

## 🔧 **Environment Setup**

Create a `.env` file in your project root:
```env
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
```

## 🚀 **Usage Examples**

### Analyze Current Directory
```bash
docker-compose up code-quality-analyzer
```

### Chat with GitHub Repository
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.env:/app/.env:ro \
  code-quality-agent \
  python -m code_quality_agent chat https://github.com/username/repo --enhanced
```

### Analyze Specific Directory
```bash
docker run --rm \
  -v /path/to/your/code:/workspace:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.env:/app/.env:ro \
  -w /workspace \
  code-quality-agent \
  python -m code_quality_agent analyze .
```

## 🛠️ **Development Mode**

For development with live code changes:
```bash
docker-compose -f docker-compose.dev.yml up
```

## 📊 **Data Persistence**

Vector databases are stored in `./data/` and persist between container runs.

## 🔒 **Security**

- Runs as non-root user (`appuser`)
- Read-only mounts for code directories
- Isolated environment

## 🐛 **Troubleshooting**

**Permission Issues:**
```bash
sudo chown -R $USER:$USER ./data
```

**Clean Build:**
```bash
docker build --no-cache -t code-quality-agent .
```

**View Logs:**
```bash
docker-compose logs code-quality-agent
```
