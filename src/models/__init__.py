from .cnn import CNNBaseline
from .lcnn import LCNN
from .resnet import ResNet18Spoof


def build_model(model_name: str, in_channels: int = 1, num_classes: int = 2):
    model_name = model_name.lower()
    builders = {
        "cnn": CNNBaseline,
        "resnet": ResNet18Spoof,
        "lcnn": LCNN,
    }
    if model_name not in builders:
        raise ValueError(f"Unsupported model name: {model_name}")
    return builders[model_name](in_channels=in_channels, num_classes=num_classes)
