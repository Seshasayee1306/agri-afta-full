# Agri-AFTA: Microservices-based Agricultural System

Agri-AFTA is a comprehensive agricultural monitoring and diagnostic system built on a microservices architecture. It integrates IoT data (ESP32), machine learning models for crop prediction, and AI-driven disease diagnosis.

## 🚀 Project Overview

The project is divided into several independent microservices, each handling a specific domain of the application. The system is designed to be deployed on Kubernetes, utilizing Cloudflare Tunnels for secure external access and Nginx for efficient frontend delivery.

---

## 🛠 Architecture & Tech Stack

### 1. Frontend (`/frontend`)
- **Framework**: React (Vite)
- **Routing**: React Router DOM
- **Rendering**: React Markdown with GFM support
- **Web Server**: Nginx (used for serving the production build)
- **Connectivity**: Tunneled via **Cloudflare Tunnel** for secure, SSL-encrypted public access without opening local ports.

### 2. Backend (`/backend`)
- **Framework**: Python (Flask)
- **Machine Learning**: s
  - XGBoost, PyTorch (TabNet)s
  - Scikit-learn, Pandas, NumPy
- **Storage & Logging**: 
  - AWS S3 (for data and model logs)
  - MLflow (for model versioning and tracking)
- **Orchestration**: Integration with Kubernetes API for dynamic retraining jobs.

### 3. Disease Service (`/disease_service`)
- **Framework**: Python (Flask)
- **AI Integration**: Groq API for advanced plant disease diagnosis and recommendations.
- **Image Processing**: OpenCV, Pillow.

### 4. API Gateway (`/agri-gateway`)
- **Framework**: .NET Core
- **Purpose**: Acts as a single entry point for all client requests, routing them to the appropriate microservices.

### 5. Infrastructure & Monitoring
- **Containerization**: Docker, Docker Compose
- **Orchestration**: Kubernetes (K8s)
- **Deployment Tool**: Skaffold
- **Monitoring**: 
  - Prometheus (Metric collection)
  - Grafana (Data visualization dashboards)
- **Networking**: Nginx Ingress and Cloudflare Tunnel.

---

## 💻 How to Execute

### Prerequisites
- Docker & Docker Desktop
- Kubernetes Cluster (Minikube or Kind)
- Skaffold (Optional, for development)
- Cloudflare Tunnel (cloudflared)

### Running Locally with Docker Compose
To start the entire stack locally for development:
```bash
docker-compose up --build
```

### Deploying to Kubernetes
1. **Apply Manifests**:
   ```bash
   kubectl apply -f k8s/
   ```
2. **Using Skaffold (Recommended for Dev)**:
   ```bash
   skaffold run
   ```

### Accessing the Application
- **Frontend**: Accessible via the Cloudflare Tunnel URL configured in `k8s/cloudflared.yaml`.
- **MLflow**: `http://localhost:5000`
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000`

---

## 📦 Key Modules Used

| Service | Primary Modules/Libraries |
| :--- | :--- |
| **Frontend** | `react`, `react-router-dom`, `vite`, `nginx` |
| **Backend** | `flask`, `xgboost`, `torch`, `pytorch-tabnet`, `mlflow`, `boto3`, `kubernetes` |
| **Disease Service** | `flask`, `groq`, `opencv-python-headless`, `pillow` |
| **Gateway** | `.NET Core`, `Controllers` |
| **Hardware** | `ESP32`, `Arduino` |

---

## ☁️ Cloudflare & Nginx Integration

The frontend is served by an **Nginx** server inside a Kubernetes pod. To expose the service securely:
1. A **Cloudflare Tunnel** (`cloudflared`) is deployed within the cluster.
2. It establishes an outbound connection to Cloudflare's edge.
3. Traffic is routed from your custom domain (e.g., `agri.yourdomain.com`) directly to the `frontend-service` over port 80.

---

## 📊 Monitoring & Logging
- **MLflow**: Tracks experiments, model parameters, and metrics during retraining.
- **Prometheus**: Scrapes metrics from the backend and disease services.
- **S3 Logging**: All sensor data and model predictions are logged to AWS S3 for long-term analysis.
