from src.config import (
    LabelsConfig,
    MiscConfig,
    ModelConfig,
    OnnxConfig,
    SamplesConfig,
    TraningConfig,
)

SAMPLES = SamplesConfig()
LABELS = LabelsConfig()
MODEL = ModelConfig()
TRAINING = TraningConfig()
MISC = MiscConfig()

ONNX = OnnxConfig(
    pytorch_model_path=MODEL.save_early_stop_path,
    version=(1, 0, 1),
)
