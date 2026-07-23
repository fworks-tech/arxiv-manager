FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose port (Render sets $PORT, default 10000)
EXPOSE 10000

CMD uvicorn run:app --host 0.0.0.0 --port ${PORT:-10000}
