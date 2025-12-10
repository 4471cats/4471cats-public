# 4471cats - YOLOv11 Cat Breed Detection

This project trains a YOLOv11 model to detect and classify 12 different cat breeds from the Oxford-IIIT Pet Dataset.

## Relevant Links

- [Train Notebook](https://www.kaggle.com/code/inogai/4471-ablation-train)
- [Eval - No Fine Tune](https://www.kaggle.com/code/inogai/4471-ablation-train-no-fine-tune)
- [Test](https://www.kaggle.com/code/inogai/4471-test)
- [Weights Publication](https://www.kaggle.com/datasets/inogai/4471-weights/code)
- [Train Dataset](https://www.kaggle.com/datasets/inogai/of3t-cats-yolo)
- [Test Dataset](https://www.kaggle.com/datasets/inogai/4471-cats-wiki-images)


## Setup

1. Install dependencies:

   ```bash
   # Using uv (recommended)
   uv sync
   ```

## Datasets

### Getting and Building the Main Dataset

The main dataset is derived from the Oxford-IIIT Pet Dataset, converted to YOLO format.

1. **Download the Oxford-IIIT Pet Dataset:**

   ```bash
   uv run python scripts/01_get_main_dataset.py
   ```

   This downloads and extracts the dataset to `datasets/oxford3tpet/`.

2. **Build the YOLO-format dataset:**

   ```bash
   uv run python scripts/02_build_main_dataset.py
   ```

   This converts the dataset to YOLO format and outputs to `datasets/of3t-cats-yolo/`.

3. **Split into train/val sets:**

   ```bash
   uv run python scripts/02b_split_dataset.py
   ```

   Creates stratified train/val splits in `datasets/of3t-cats-yolo/images/{train,val}/`.

### Building the Wiki Dataset (Supplementary Images)

Additional images can be collected from Wikimedia Commons:

```bash
uv run python scripts/03_build_wiki_dataset.py
```

This downloads images listed in `scripts/wiki_sources.csv` to `datasets/4471-cats-wiki-images/`.

## Backend API

The project includes a FastAPI-based prediction server for real-time cat breed detection.

### Running the API Server

```bash
uv run python scripts/21_predict_video.py serve --model weights/best.pt
```

**Options:**

- `-m, --model`: Path to YOLO weights file (default: `weights/best.pt`)
- `--host`: Host address (default: `0.0.0.0`)
- `-p, --port`: Port number (default: `8000`)
- `-c, --conf`: Confidence threshold (default: `0.25`)

### API Endpoints

- `GET /api/classes` - Get list of supported cat breeds
- `POST /api/predict` - Upload an image for prediction
- `WebSocket /api/ws` - Real-time video frame prediction

### Downloading Weights

To download pre-trained weights from Kaggle:

```bash
uv run python scripts/21_predict_video.py download-weights
```

### Local Detection CLI

Run detection on local video files:

```bash
uv run python scripts/21_predict_video.py detect -m weights/best.pt -s video.mp4
```

## Project Structure

```text
4471cats/
├── scripts/
│   ├── 01_get_main_dataset.py   # Download Oxford-IIIT Pet Dataset
│   ├── 02_build_main_dataset.py # Convert to YOLO format
│   ├── 02b_split_dataset.py     # Create train/val splits
│   ├── 03_build_wiki_dataset.py # Download wiki images
│   ├── 11_upload_main_dataset.py # Upload main dataset to Kaggle
│   ├── 12_upload_wiki_dataset.py # Upload wiki dataset to Kaggle
│   └── 21_predict_video.py      # API server and detection CLI
├── datasets/
│   ├── oxford3tpet/             # Raw Oxford dataset
│   ├── of3t-cats-yolo/          # YOLO-format dataset
│   └── 4471-cats-wiki-images/   # Wiki images
├── weights/                      # Model weights
├── notebooks/                    # Training and testing notebooks
└── README.md
```