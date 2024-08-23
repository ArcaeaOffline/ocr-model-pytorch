import numpy as np
import onnxruntime as onnxrt
import torch
from PIL import Image
from rich.console import Console
from rich.text import Text

import config as cfg
from src.models import CRNN

console = Console()


def decode_predictions(
    classes: list[str],
    predictions: np.ndarray,
    blank_token: str = cfg.MODEL.blank_token,
):
    texts = []
    for i in range(predictions.shape[0]):
        string = ""
        batch_e = predictions[i]

        for j in range(len(batch_e)):
            string += classes[batch_e[j]]

        string = string.split(blank_token)
        string = [x for x in string if x != ""]
        string = [list(set(x))[0] for x in string]
        texts.append("".join(string))
    return texts


class OnnxOutputValidator:
    def __init__(self, model: CRNN, onnx_patched_model_path: str, classes: list[str]):
        self.classes = classes
        self.model = model
        self.onnx_session = onnxrt.InferenceSession(
            onnx_patched_model_path, providers=onnxrt.get_available_providers()
        )

    def __resize_image(self, image: Image.Image):
        return image.resize(
            (cfg.MODEL.image_width, cfg.MODEL.image_height),
            resample=Image.Resampling.BILINEAR,
        )

    def onnx_input(self, image: Image.Image):
        image = self.__resize_image(image)
        np_image = np.array(image)
        # The following commented line is included in the patched model file
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # image = image[None, ...]
        return np_image

    def torch_input(self, image: Image.Image):
        image = self.__resize_image(image)
        np_image = np.array(image)
        np_image = np.transpose(np_image, (2, 0, 1)).astype(np.float32)
        np_image = np_image[None, ...]
        return torch.from_numpy(np_image).float()

    def onnx_outputs(self, image: Image.Image):
        """tuple[model_output, decoded_output]"""
        inputs = {"raw_image": self.onnx_input(image)}

        outputs = self.onnx_session.run(["model_output", "decoded_output"], inputs)
        return outputs

    def torch_outputs(self, image: Image.Image):
        """tuple[model_output, decoded_output]"""
        model_outputs, _ = self.model(self.torch_input(image))
        decoded_outputs = model_outputs.permute(1, 0, 2)
        decoded_outputs = torch.softmax(decoded_outputs, 2)
        decoded_outputs = torch.argmax(decoded_outputs, 2)
        return model_outputs, decoded_outputs

    def __assert_close(self, torch_results, onnx_results):
        assert len(torch_results) == len(onnx_results)
        for _torch, _onnx in zip(torch_results, onnx_results):
            torch.testing.assert_close(_torch, torch.tensor(_onnx))

    def assert_onnx_torch_close(self, image: Image.Image):
        onnx_model_outputs, onnx_decoded_outputs = self.onnx_outputs(image)
        torch_model_outputs, torch_decoded_outputs = self.torch_outputs(image)

        try:
            self.__assert_close(torch_model_outputs, onnx_model_outputs)
            console.print("Model outputs match.", style="green")
        except AssertionError:
            console.print("Model outputs mismatch!", style="red")
            console.print_exception()

        try:
            self.__assert_close(torch_decoded_outputs, onnx_decoded_outputs)
            console.print("Decode outputs match.", style="green")
        except AssertionError:
            console.print("Decode outputs mismatch!", style="red")
            console.print_exception()

    def onnx_predictions(self, image: Image.Image):
        return decode_predictions(self.classes, np.array(self.onnx_outputs(image)[1]))

    def torch_predictions(self, image: Image.Image):
        return decode_predictions(self.classes, self.torch_outputs(image)[1])

    def answer_readable(self, predictions: list[str]):
        output = ""
        last_char = None
        for char in predictions:
            if char == last_char:
                continue
            if char == cfg.MODEL.blank_token:
                last_char = None
                continue
            output += char
            last_char = char
        return output

    def print_answers(self, image_path: str):
        console.print(image_path)
        image = Image.open(image_path)

        text = Text(" ")
        text.append(
            f"torch: {self.answer_readable(self.torch_predictions(image))}", "orange3"
        )
        text.append(" | ")
        text.append(
            f"ONNX: {self.answer_readable(self.onnx_predictions(image))}",
            "dodger_blue1",
        )
        console.print(text)
