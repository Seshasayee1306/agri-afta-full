FROM python:3.10-slim

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
# Copy backend (includes retrain/)
# -----------------------------
COPY backend /app/backend
COPY fed_afta /app/fed_afta
COPY dataset /app/dataset

# ❌ REMOVE THIS (WRONG)
# COPY retrain /app/retrain

# -----------------------------
# Ensure backend is a package
# -----------------------------
RUN test -f /app/backend/__init__.py

EXPOSE 8000

CMD ["python", "-m", "backend.app"]
