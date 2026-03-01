# S3 Logging + Retraining (Local & Docker)

## What gets uploaded to S3

If `S3_BUCKET` is set, the backend uploads:

- `predictions/*.json`: every `/predict_full_intelligent` request payload + response + filled context
- `labeled/*.json`: every `/label` submission (12 AFTA features + label)

Only **labeled** objects are used for retraining.

## Required environment variables

- `S3_BUCKET` (required to enable S3 logging)
- `S3_PREFIX` (optional, default `agri`)
- `AWS_REGION` (recommended)
- AWS credentials:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_SESSION_TOKEN` (optional)

You can also use IAM roles in cloud environments (EKS/ECS), in which case you typically do not set access keys.

## Local run (no Docker)

From repo root:

```bash
export S3_BUCKET=your-bucket
export S3_PREFIX=agri
export AWS_REGION=ap-south-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

python3 -m backend.app
```

Send a prediction (uploads `predictions/*.json`):

```bash
curl -s -X POST http://127.0.0.1:8000/predict_full_intelligent \
  -H 'Content-Type: application/json' \
  -d '{"sowing_date":"2024-01-01","current_date":"2024-02-01","soil_moisture":40,"temperature":25,"humidity":60,"ph":6.5,"region":"South India","crop_type":"Rice","soil_type":"clay"}'
```

Send a label/feedback (uploads `labeled/*.json`):

```bash
curl -s -X POST http://127.0.0.1:8000/label \
  -H 'Content-Type: application/json' \
  -d '{"sowing_date":"2024-01-01","current_date":"2024-02-01","soil_moisture":40,"temperature":25,"humidity":60,"ph":6.5,"region":"South India","crop_type":"Rice","soil_type":"clay","label":1}'
```

## Docker

Set env vars in your shell (or a `.env` file used by docker compose), then:

```bash
docker compose up --build backend
```

## Retraining job

The retraining script `backend/retrain/retrain.py`:
- loads the base dataset from `BASE_TRAINING_DATASET` (default `/app/dataset/irrigation_dataset.csv`)
- downloads newly added labeled rows from S3 under `${S3_PREFIX}/labeled/`
- appends them and runs federated retraining
- writes the updated model to `backend/final_model.pkl`
- uploads the model to:
  - `${S3_PREFIX}/models/final_model_<timestamp>.pkl`
  - `${S3_PREFIX}/models/final_model_latest.pkl`

Run it locally:

```bash
python3 -m backend.retrain.retrain
```

In Docker (inside the backend container):

```bash
docker compose exec backend python -m backend.retrain.retrain
```

