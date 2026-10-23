"""MicroNet-M0 transfer learning with pinned author weights (see _micronet/LICENSE)."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from ._micronet.network import MicroNet

WEIGHTS_SHA256 = "987159bcdd92ef2792da9dc4e8e698e8065a6aa0899e31d9762080c621d81b61"
WEIGHTS_FILENAME = "micronet-m0-987159bc.pth"
# The author's published endpoint is HTTP; digest pinning rejects modified downloads.
WEIGHTS_URL = "http://www.svcl.ucsd.edu/projects/micronet/assets/micronet-m0.pth"


def m0_config():
    """Values from the author's eval_micronet_m0.sh, without a yacs dependency."""
    return SimpleNamespace(MODEL=SimpleNamespace(
        MICRONETS=SimpleNamespace(
            NET_CONFIG="msnx_dy6_exp4_4M_221", STEM_CH=4, STEM_GROUPS=(2, 2),
            STEM_DILATION=1, STEM_MODE="spatialsepsf", OUT_CH=640,
            DEPTHSEP=True, SHUFFLE=True, POINTWISE="group", DROPOUT=0.05,
        ),
        ACTIVATION=SimpleNamespace(
            MODULE="DYShiftMax", ACT_MAX=2.0, LINEARSE_BIAS=False,
            INIT_A_BLOCK3=(1.0, 0.0), INIT_A=(1.0, 1.0), INIT_B=(0.0, 0.0), REDUCTION=8,
        ),
    ))


class SingletonBatchNorm1d(nn.BatchNorm1d):
    def forward(self, inputs):
        # Keep the final singleton batch instead of discarding a training sample.
        if self.training and inputs.shape[0] == 1:
            return F.batch_norm(inputs, self.running_mean, self.running_var,
                                self.weight, self.bias, training=False, eps=self.eps)
        return super().forward(inputs)


def _load_pretrained_state():
    path = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS_FILENAME
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL, str(path), hash_prefix=WEIGHTS_SHA256)
    with path.open("rb") as stream:
        actual = hashlib.sha256(stream.read()).hexdigest()
    if actual != WEIGHTS_SHA256:
        raise RuntimeError(f"MicroNet-M0 pretrained SHA256 mismatch: {path}. Remove the corrupt cache and retry.")
    return torch.load(path, map_location="cpu", weights_only=True)


def create_micronet_m0(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = MicroNet(m0_config())
    model.classifier[0].linear[1] = SingletonBatchNorm1d(640)
    if pretrained:
        state = _load_pretrained_state()
        model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    # The project uses unactivated logits with CE, not the original ImageNet BN/activation head.
    model.classifier[-1] = nn.Linear(640, num_classes)
    return model
