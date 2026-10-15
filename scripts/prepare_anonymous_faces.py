"""Prepare the 10 A/B classroom images for anonymous manual face review."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anonymous_faces import run_batch

if __name__ == '__main__':
    run_batch()
