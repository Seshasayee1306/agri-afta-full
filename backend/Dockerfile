FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

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

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
