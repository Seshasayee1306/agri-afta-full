FROM python:3.10-slim

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install dependencies with PyTorch CPU-only to save space
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    rm -rf /root/.cache/pip && \
    pip install --no-cache-dir flask flask-cors python-dotenv groq && \
    rm -rf /root/.cache/pip && \
    pip install --no-cache-dir numpy pandas scikit-learn xgboost && \
    rm -rf /root/.cache/pip && \
    pip install --no-cache-dir pytorch-tabnet shap joblib && \
    rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

EXPOSE 8000

# Use Flask
CMD ["python", "app.py"]