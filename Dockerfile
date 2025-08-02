# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set working directory
WORKDIR /app

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    g++ \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04 already has proper Python setup, no symlinks needed

# Install Rust for indexing
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install transformers==4.40.2 flask flask-restx pybind11

# Copy the package code
COPY pkg/ ./pkg/

# Compile the C++ engine
RUN cd pkg && \
    c++ -std=c++20 -O3 -shared -fPIC $(python3 -m pybind11 --includes) infini_gram/cpp_engine.cpp -o infini_gram/cpp_engine$(python3-config --extension-suffix)

# Compile the Rust indexing code
RUN cd pkg && \
    cargo build --release && \
    mv target/release/rust_indexing infini_gram/

# Copy API server code
COPY api/ ./api/

# Set environment variables
ENV PYTHONPATH=/app/pkg

# Create data directory and log directory
RUN mkdir -p /data && \
    mkdir -p /home/ubuntu/logs && \
    touch /home/ubuntu/logs/flask_api.log && \
    chmod 777 /home/ubuntu/logs && \
    chmod 666 /home/ubuntu/logs/flask_api.log

# Add health check endpoint to the API server
RUN echo 'import sys; sys.path.append("../pkg"); from infini_gram.engine import InfiniGramEngine' > /tmp/health_check.py

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:5000/health', timeout=5)" || exit 1

# Run the application
CMD ["python3", "api/api_server.py", "--FLASK_PORT=5000", "--MODE=api", "--CONFIG_FILE=/app/api_config.json"] 