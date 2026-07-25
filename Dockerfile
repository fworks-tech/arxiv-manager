FROM python:3.11

WORKDIR /app

# Install system dependencies for sentence-transformers / PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy app
COPY . .

# Expose port
EXPOSE 10000

CMD uvicorn run:app --host 0.0.0.0 --port ${PORT:-10000}
