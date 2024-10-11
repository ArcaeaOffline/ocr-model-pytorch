"""
A quick glance of the inference results of your training samples.
"""

import json

import torch
from rich.progress import Progress

import config as cfg
from inference import inference
from src.models import CRNN
from src.utils import DatabaseHelper, SamplesHelper
from src.utils.model_info import ModelInfo

database_helper = DatabaseHelper()
samples_helper = SamplesHelper()

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

model_filepath = cfg.MODEL.save_early_stop_path
model.load_state_dict(torch.load(model_filepath, weights_only=True))
model.eval()

with Progress() as progress:
    samples = samples_helper.samples

    task = progress.add_task("Evaluating...", total=len(samples))

    evaluated_count = 0
    error_count = 0

    for sample in samples:
        progress.advance(task)

        label = database_helper.get_sample_label(sample)

        if label is None:
            continue

        model_label = inference(sample, model, "cpu")
        model_label = "".join(s.rstrip(cfg.MODEL.pad_token) for s in model_label)

        evaluated_count += 1

        if label != model_label:
            error_count += 1
            progress.console.print(
                f"[red]![/red] {sample.name} | expected {label} | got {model_label}"
            )

    percentage = (error_count / evaluated_count) if evaluated_count > 0 else 0
    percentage *= 100
    progress.console.print(
        f"[cyan]i[/cyan] Overall errors {error_count}/{evaluated_count} ({percentage:.2f}%)"
    )
