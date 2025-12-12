FROM python:3.10-slim

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install dependencies with cleanup in single layer to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

EXPOSE 8000

# Use Flask (not uvicorn)
CMD ["python", "app.py"]