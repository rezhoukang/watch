"""Project directories and supported media formats shared by backend modules."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "frontend"
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
RESULTS_DIR = BASE_DIR / "data" / "results"
VIDEO_DIR = BASE_DIR / "data" / "videos"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
DATASETS_DIR = BASE_DIR / "data" / "datasets"
RUNS_DIR = BASE_DIR / "data" / "runs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
