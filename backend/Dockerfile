FROM python:3.10-slim

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Install dependencies
# -----------------------------
COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir flask flask-cors python-dotenv groq && \
    pip install --no-cache-dir numpy pandas scikit-learn xgboost && \
    pip install --no-cache-dir pytorch-tabnet shap joblib scipy && \
    rm -rf /root/.cache/pip

# -----------------------------
# Copy application code (CRITICAL)
# -----------------------------
COPY backend /app/backend
COPY fed_afta /app/fed_afta
COPY dataset /app/dataset
COPY retrain /app/retrain

# -----------------------------
# Ensure backend is a package
# -----------------------------
RUN test -f /app/backend/__init__.py

# -----------------------------
# Expose port
# -----------------------------
EXPOSE 8000

# -----------------------------
# ✅ Correct module execution
# -----------------------------
CMD ["python", "-m", "backend.app"]
