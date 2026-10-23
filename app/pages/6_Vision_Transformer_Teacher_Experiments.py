"""teacher_vision_transformer experiment launcher."""

from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from standard_launcher import render_standard_launcher

render_standard_launcher(
    "teacher_vision_transformer",
    ["appleleaf9", "plantvillage", "plantpathology2021"],
    ["swin_v2_t","maxvit_t","vit_b_16","dinov2_vitb14"],
    unified=globals().get("UNIFIED_LAUNCHER", False),
)
