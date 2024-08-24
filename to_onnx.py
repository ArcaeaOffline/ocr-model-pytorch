import json

import onnx
import torch
from PIL import Image
from rich.console import Console

import config as cfg
from src.models import CRNN
from src.to_onnx.patch import patch_onnx_model, write_onnx_custom_metadata
from src.to_onnx.validator import OnnxOutputValidator
from src.utils.model_info import ModelInfo

console = Console()

model_filepath = cfg.ONNX.pytorch_model_path
if not model_filepath.exists():
    raise FileNotFoundError(f"Model file not found at {model_filepath}")


model_info = ModelInfo.from_dict(
    json.loads(cfg.MODEL.save_info_path.read_text(encoding="utf-8"))
)
labels = model_info.labels
num_chars = len(labels) - 1

model = CRNN(
    resolution=(cfg.MODEL.image_width, cfg.MODEL.image_height),
    dims=cfg.MODEL.dims,
    num_chars=num_chars,
    use_attention=cfg.MODEL.use_attention,
    use_ctc=cfg.MODEL.use_ctc,
    grayscale=cfg.MODEL.grayscale,
)

model.load_state_dict(torch.load(model_filepath))
model.eval()

torch_input = torch.randn(
    1, 1 if cfg.MODEL.grayscale else 3, cfg.MODEL.image_height, cfg.MODEL.image_width
)

onnx_filepath = model_filepath.with_suffix(".onnx")
onnx_patched_filepath = onnx_filepath.with_stem(onnx_filepath.stem + "_patched")


def convert():
    torch.onnx.export(
        model,
        torch_input,
        onnx_filepath,
        input_names=["image"],
        output_names=["model_output"],
    )

    onnx_model = onnx.load(onnx_filepath)
    patch_onnx_model(onnx_model)
    write_onnx_custom_metadata(
        onnx_model,
        version=cfg.ONNX.version,
        built_timestamp=model_info.built_timestamp,
    )
    onnx.save(onnx_model, onnx_patched_filepath)


def validate():
    # you may adjust these files with your choices

    image_path = "examples/t1_score_9586884.png"
    image = Image.open(image_path)
    validator = OnnxOutputValidator(
        model, str(onnx_patched_filepath.resolve()), labels
    )
    validator.assert_onnx_torch_close(image)
    validator.print_answers(image_path)

    validator.print_answers("examples/t1_far_5.png")
    validator.print_answers("examples/t1_lost_8.png")
    validator.print_answers("examples/t2_far_7.png")
    validator.print_answers("examples/t2_pure_798.png")
    validator.print_answers("examples/t2_pure_1093.png")


convert()
validate()
