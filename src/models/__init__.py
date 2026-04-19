from .cnn import CNNBaseline
from .lcnn import LCNN
from .resnet import ResNet18Spoof


MODEL_NAME_ALIASES = {
    "cnn": "cnn",
    "lcnn": "lcnn",
    "resnet": "resnet18",
    "resnet18": "resnet18",
}


def canonicalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized not in MODEL_NAME_ALIASES:
        raise ValueError(f"Unsupported model name: {model_name}")
    return MODEL_NAME_ALIASES[normalized]


def build_model(model_name: str, in_channels: int = 1, num_classes: int = 2):
    canonical_name = canonicalize_model_name(model_name)
    builders = {
        "cnn": CNNBaseline,
        "resnet18": ResNet18Spoof,
        "lcnn": LCNN,
    }
    return builders[canonical_name](in_channels=in_channels, num_classes=num_classes)
