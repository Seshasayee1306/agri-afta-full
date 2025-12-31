FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (better caching)
COPY backend/requirements.txt ./requirements.txt

# Install dependencies EXACTLY as pinned
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy application code
COPY backend /app/backend
COPY fed_afta /app/fed_afta
COPY dataset /app/dataset

# Sanity check
RUN test -f /app/backend/__init__.py

EXPOSE 8000
CMD ["python", "-m", "backend.app"]
