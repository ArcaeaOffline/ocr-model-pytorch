import json

import numpy as np
import torch
from PIL import Image

import config as cfg
from src.models import CRNN
from src.utils.model_decoders import decode_padded_predictions, decode_predictions
from src.utils.model_info import ModelInfo

model_info = ModelInfo.from_dict(
    json.loads(cfg.MODEL.save_info_path.read_text(encoding="utf-8"))
)
labels = model_info.labels


def inference(image_path):
    # Hardcoded resize
    image = Image.open(image_path).convert("RGB")
    image = image.resize((250, 50), resample=Image.Resampling.BILINEAR)
    image = np.array(image)
    print(image.shape)

    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = image[None, ...]
    image = torch.from_numpy(image)
    if str(device) == "cuda":
        image = image.cuda()
    image = image.float()
    with torch.no_grad():
        preds, _ = model(image)

    if model.use_ctc:
        answer = decode_predictions(preds, labels)
    else:
        answer = decode_padded_predictions(preds, labels)
    return answer


if __name__ == "__main__":
    # Setup model and load weights
    model = CRNN(
        dims=cfg.MODEL.dims,
        num_chars=len(labels) - 1,
        use_attention=cfg.MODEL.use_attention,
        use_ctc=cfg.MODEL.use_ctc,
        grayscale=cfg.MODEL.grayscale,
    )
    device = torch.device("cpu")
    model.to(device)
    model.load_state_dict(cfg.MODEL.save_early_stop_path)
    model.eval()

    filepath = "test.jpg"  # Replace with your image!
    answer = inference(filepath)
    print(f"text: {answer}")
