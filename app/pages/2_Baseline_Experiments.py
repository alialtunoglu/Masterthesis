"""baseline experiment launcher."""

from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.standard_launcher import render_standard_launcher
from src.models.model_catalog import STUDENT_CNN_MODELS

render_standard_launcher(
    "baseline",
    ["appleleaf9", "plantvillage", "plantpathology2021"],
    ["mobilenet_v3_small", *sorted(STUDENT_CNN_MODELS - {"mobilenet_v3_small"})],
    unified=globals().get("UNIFIED_LAUNCHER", False),
)
