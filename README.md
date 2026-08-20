# ☕ BeanSight - Coffee Bean Analyzer

**AI-powered coffee bean classification and defect detection** using dual EfficientNetB0 models with ONNX-optimized inference, served via FastAPI on Google Cloud Run.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.18-purple)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![Cloud Run](https://img.shields.io/badge/Cloud_Run-Deployed-4285F4?logo=googlecloud)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Google Cloud Run                        │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Production Container (~300MB)                     │  │
│  │  FastAPI → ONNX Runtime → roast_model.onnx        │  │
│  │                         → defect_model.onnx       │  │
│  └────────┬─────────────────────┬────────────────────┘  │
│           │                     │                        │
│     Supabase PostgreSQL   Supabase Storage               │
└──────────────────────────────────────────────────────────┘
```

### Two Models, One Pipeline

| Model | Classes | Dataset |
|-------|---------|---------|
| **Roast Classifier** | Dark, Green, Light, Medium (4) | [Coffee Bean Dataset Resized](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224) |
| **Defect Detector** | Broken, Cut, Dry Cherry, + 14 more (17) | [Coffee Green Bean with 17 Defects](https://www.kaggle.com/datasets/sujitraarw/coffee-green-bean-with-17-defects-original) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- Kaggle API credentials (for dataset download)

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/coffee-bean-analyzer.git
cd coffee-bean-analyzer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install production deps (for running the web app)
pip install -r requirements.txt

# Install training deps (for training models)
pip install -r requirements-train.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Download & Prepare Data

```bash
# Set Kaggle credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Download both datasets and create 80/10/10 splits
python data/download_dataset.py --task both
```

### 4. Train Models

```bash
# Train both models (roast + defect)
python -m model.train --task both

# Or train individually
python -m model.train --task roast --data-dir data/roast
python -m model.train --task defect --data-dir data/defect
```

Training outputs:
- `checkpoints/roast_best.pt` — PyTorch checkpoint
- `checkpoints/roast_model.onnx` — ONNX export (for production)
- `checkpoints/defect_best.pt` — PyTorch checkpoint
- `checkpoints/defect_model.onnx` — ONNX export (for production)

### 5. Run the Web App

```bash
# Development mode
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# Open http://localhost:8080
```

---

## 🐳 Docker

### Production Image (ONNX-only, ~300MB)

```bash
# Build
docker build -t coffee-bean-analyzer .

# Run
docker run -p 8080:8080 --env-file .env coffee-bean-analyzer
```

### Training Image

```bash
# Build training image
docker build -f Dockerfile.train -t coffee-bean-trainer .

# Run training
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  coffee-bean-trainer
```

### Docker Compose

```bash
# Run web app
docker compose up app

# Run training (separate profile)
docker compose --profile training up train
```

---

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest httpx

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=html
```

---

## ☁️ Cloud Deployment (Google Cloud Run)

### 1. Set Up Google Cloud

```bash
# Install gcloud CLI, authenticate, and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# Create Artifact Registry repo
gcloud artifacts repositories create coffee-bean-analyzer \
  --repository-format=docker \
  --location=us-central1
```

### 2. Set Up Supabase

1. Create a project at [supabase.com](https://supabase.com)
2. Copy the database connection string (pooled, port 6543)
3. Create a storage bucket named `coffee-uploads`
4. Copy the project URL and service role key

### 3. Deploy

```bash
# Build and push to Artifact Registry
IMAGE=us-central1-docker.pkg.dev/YOUR_PROJECT/coffee-bean-analyzer/app:latest
docker build -t $IMAGE .
docker push $IMAGE

# Deploy to Cloud Run
gcloud run deploy coffee-bean-analyzer \
  --image $IMAGE \
  --platform managed \
  --region us-central1 \
  --cpu-boost \
  --memory 2Gi \
  --min-instances 0 \
  --max-instances 3 \
  --allow-unauthenticated \
  --set-env-vars "ENVIRONMENT=production,DATABASE_URL=your_db_url,SUPABASE_URL=your_url,SUPABASE_SERVICE_KEY=your_key"
```

### 4. CI/CD (GitHub Actions)

The included `.github/workflows/deploy.yml` automatically:
1. Runs `pytest` on every push/PR
2. Builds the Docker image on push to `main`
3. Pushes to Artifact Registry
4. Deploys to Cloud Run

Required GitHub Secrets:
- `GCP_PROJECT_ID`
- `WIF_PROVIDER`
- `WIF_SERVICE_ACCOUNT`

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Upload image → dual prediction (roast + defect) |
| `GET` | `/api/history` | Recent predictions (paginated) |
| `GET` | `/api/stats` | Aggregate statistics |
| `GET` | `/health` | Health check with model status |

### `POST /api/predict`

```bash
curl -X POST http://localhost:8080/api/predict \
  -F "file=@bean_photo.jpg"
```

Response:
```json
{
  "roast": {
    "prediction": "Medium",
    "confidence": 0.94,
    "probabilities": {"Dark": 0.02, "Green": 0.01, "Light": 0.03, "Medium": 0.94}
  },
  "defect": {
    "prediction": "Fungus Damage",
    "confidence": 0.78,
    "probabilities": {"Broken": 0.01, "...": "...", "Withered": 0.03}
  },
  "inference_time_ms": 52.3,
  "image_url": "https://xxx.supabase.co/storage/v1/object/public/coffee-uploads/img.jpg"
}
```

---

## 📁 Project Structure

```
coffee-bean-analyzer/
├── app/                     # FastAPI web application
│   ├── main.py              # Routes, middleware, lifespan
│   ├── config.py            # Environment configuration
│   ├── database.py          # Async PostgreSQL layer
│   ├── storage.py           # Supabase Storage client
│   └── static/              # Frontend SPA
├── model/                   # ML pipeline
│   ├── network.py           # EfficientNetB0 architecture
│   ├── dataset.py           # PyTorch Dataset + augmentation
│   ├── train.py             # Two-phase training + ONNX export
│   └── predict.py           # ONNX Runtime inference
├── data/                    # Datasets
│   └── download_dataset.py  # Kaggle download + split
├── tests/                   # Automated tests
│   └── test_api.py          # pytest endpoint tests
├── checkpoints/             # Model weights (.onnx)
├── .github/workflows/       # CI/CD pipeline
├── Dockerfile               # Production image (~300MB)
├── Dockerfile.train         # Training image (PyTorch)
├── docker-compose.yml       # Local development
├── requirements.txt         # Production deps
└── requirements-train.txt   # Training deps
```

---

## 📄 License

MIT License
