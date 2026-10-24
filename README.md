# Chest CT Scan Classification — End-to-End MLOps Pipeline

Classifies chest CT scan images as **adenocarcinoma** (a type of lung cancer) or **normal** using transfer learning on VGG16. The project is built as a reproducible pipeline: DVC runs each stage, MLflow (on DagsHub) tracks experiments, and a Flask app serves predictions.

> Built while learning MLOps (based on a guided course project). I later found and fixed a data-leakage bug in the evaluation stage — details below.

## Results

| Metric (validation set) | Score |
|---|---|
| Accuracy | _update after `dvc repro`_ |
| Loss | _update after `dvc repro`_ |

The numbers are in `scores.json` after running the pipeline.

### Bug I found and fixed: evaluation data leakage

The first version reported **100% accuracy**, which was too good to be true.

- Training used `validation_split=0.20` → images 0–20% of each class were validation, 20–100% were training.
- Evaluation used `validation_split=0.30` → it evaluated on images 0–30%.
- So **one third of the "evaluation" images had been seen during training.**

Fix: one shared constant `VALIDATION_SPLIT` in `src/Cnnclassifier/constants/__init__.py`, used by both training and evaluation. Lesson: the train/validation split must be defined in exactly one place.

## Pipeline

```text
Stage 1: Data Ingestion      download + unzip CT images (Google Drive → artifacts/)
Stage 2: Prepare Base Model  VGG16 (ImageNet weights, frozen) + new softmax head
Stage 3: Model Training      augmentation, 80/20 split, save trained_model.h5
Stage 4: Evaluation          loss + accuracy → scores.json, logged to MLflow
```

Run all stages (only changed stages re-run):

```bash
dvc repro
dvc dag      # show the stage graph
```

## Model

| Setting | Value |
|---|---|
| Base model | VGG16, ImageNet weights, top removed, base frozen |
| Input size | 224 × 224 × 3 |
| Head | Flatten → Dropout(0.5) → Dense(2, softmax) |
| Optimizer | Adam, learning rate 0.01 |
| Epochs / batch size | 5 / 16 |
| Augmentation | rotation, flip, shift, shear, zoom |

All hyperparameters are in `params.yaml`.

## Project structure

```text
config/config.yaml          paths for every stage
params.yaml                 hyperparameters
dvc.yaml                    pipeline stage definitions
src/Cnnclassifier/
├── components/             ingestion, base model, training, evaluation
├── config/configuration.py reads YAML → typed config objects
├── entity/                 config dataclasses
├── pipelines/              stage_01 … stage_04 + prediction
└── utils/                  YAML/JSON helpers, base64 image decoding
research/                   notebooks used to prototype each stage
app.py                      Flask app: upload an image → prediction
```

## Run it

```bash
git clone https://github.com/QosainBukhari/Chest_CT_Scan_Classification_End_to_end_Project_MLOPS.git
cd Chest_CT_Scan_Classification_End_to_end_Project_MLOPS
conda create -n chest python=3.11 -y
conda activate chest
pip install -r requirements.txt
```

MLflow tracking on DagsHub (optional) — set these as environment variables, never in code:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<token>
```

Train and start the app:

```bash
dvc repro
python app.py        # open http://localhost:8080
```

## Tech stack

Python · TensorFlow/Keras · VGG16 · DVC · MLflow · DagsHub · Flask

## Limitations

- Small dataset and only two classes; this is a learning project, **not** a medical tool.
- No test set separate from validation yet; next step is a held-out test split and a confusion matrix.
- Grad-CAM heatmaps (to show which region drives the prediction) are planned.
