FROM python:3.10-slim

WORKDIR /app

# Copy requirements first
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    rm -rf /root/.cache/pip && \
    pip install --no-cache-dir flask flask-cors python-dotenv groq && \
    rm -rf /root/.cache/pip && \
    pip install --no-cache-dir numpy pandas scikit-learn xgboost && \
    rm -rf /root/.cache/pip && \
    pip install --no-cache-dir pytorch-tabnet shap joblib scipy && \
    rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy fed_afta module from root
COPY fed_afta ./fed_afta

# Copy dataset
COPY dataset ./dataset

COPY retrain ./retrain

# Copy backend code (includes retrain/ directory)
COPY backend .

EXPOSE 8000

CMD ["python", "app.py"]