"""Lightweight transformer student baselines, separate from KD."""

from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.standard_launcher import render_standard_launcher
from src.models.model_catalog import STUDENT_VIT_MODELS

render_standard_launcher(
    "baseline", ["appleleaf9", "plantvillage", "plantpathology2021"],
    sorted(STUDENT_VIT_MODELS), unified=globals().get("UNIFIED_LAUNCHER", False),
)
