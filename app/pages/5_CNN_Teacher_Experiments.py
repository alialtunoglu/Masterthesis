"""teacher_cnn experiment launcher."""

from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from standard_launcher import render_standard_launcher

render_standard_launcher(
    "teacher_cnn",
    ["appleleaf9", "plantvillage", "plantpathology2021"],
    ["resnet50","resnet101","densenet121","densenet201","vgg19_bn","efficientnet_b3","efficientnet_b4","convnext_tiny","convnext_base","regnet_y_8gf"],
    unified=globals().get("UNIFIED_LAUNCHER", False),
)
