# CodeAnalyzer Pro - Enhanced Code-Aware RAG System
# Multi-stage Docker build for optimized image size and security
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimization
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies (minimal)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy the application code
COPY . .

# Create data directories for Enhanced RAG system
RUN mkdir -p data/vector_db data/cache data/logs

# Set environment variables for Enhanced Code-Aware RAG optimization
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Memory optimization for vector operations
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
# Enhanced RAG specific optimizations
ENV CHROMA_DB_PATH=/app/data/vector_db
ENV SENTENCE_TRANSFORMERS_CACHE=/app/data/cache
ENV TRANSFORMERS_CACHE=/app/data/cache
ENV HF_HOME=/app/data/cache

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check for Enhanced RAG system
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import code_quality_agent; print('Enhanced RAG system ready')" || exit 1

# Expose port (for future web interface)
EXPOSE 8000

# Default command - Enhanced Code-Aware RAG system
CMD ["python", "-m", "code_quality_agent", "--help"]
