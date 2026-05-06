# Disease Classifier Training (Kaggle PlantVillage)

This project can train a real disease classifier from:
https://www.kaggle.com/datasets/emmarex/plantdisease

## 1) Download and extract dataset

```bash
# Requires Kaggle API configured (~/.kaggle/kaggle.json)
kaggle datasets download -d emmarex/plantdisease -p /tmp
unzip -o /tmp/plantdisease.zip -d /tmp/plantdisease
```

Find the folder that contains class subfolders (ImageFolder format), e.g.:
`/tmp/plantdisease/PlantVillage`

## 2) Train model

From repo root:

```bash
python3 -m backend.disease.train_classifier \
  --data-dir /tmp/plantdisease/PlantVillage \
  --epochs 8 \
  --batch-size 32 \
  --out-model backend/disease_model.pt \
  --out-classes backend/disease_class_names.json
```

## 3) Run backend

`/predict_disease` will automatically use the trained model when both files exist:
- `backend/disease_model.pt`
- `backend/disease_class_names.json`

If these files are missing, backend falls back to heuristic segmentation-based labels.

Optional env overrides:
- `DISEASE_MODEL_PATH`
- `DISEASE_CLASSES_PATH`
