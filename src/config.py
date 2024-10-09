import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from peewee import SqliteDatabase


@dataclasses.dataclass
class SamplesConfig:
    path: Path = Path("samples")
    globs: list[str] = dataclasses.field(default_factory=lambda: ["*.jpg", "*.png"])


@dataclasses.dataclass
class LabelsConfig:
    """
    :param max_width: 0 for auto detection
    """

    max_width: int = 0
    database = SqliteDatabase("labels.db")


@dataclasses.dataclass
class ModelConfig:
    image_width: int = 220
    image_height: int = 50
    grayscale: bool = False
    dims: int = 256

    use_attention: bool = True
    use_ctc: bool = True

    blank_token: str = "∅"
    pad_token: str = "-"

    save_base_path: Path = Path("outputs") / "model.pth"
    save_early_stop_path_override: Optional[Path] = None
    save_best_acc_path_override: Optional[Path] = None
    save_info_path_override: Optional[Path] = None

    @property
    def save_early_stop_path(self):
        if self.save_early_stop_path_override is not None:
            return self.save_early_stop_path_override

        return self.save_base_path.with_stem(self.save_base_path.stem + "-early_stop")

    @property
    def save_best_acc_path(self):
        if self.save_best_acc_path_override is not None:
            return self.save_best_acc_path_override

        return self.save_base_path.with_stem(self.save_base_path.stem + "-best_acc")

    @property
    def save_info_path(self):
        if self.save_info_path_override is not None:
            return self.save_info_path_override

        return self.save_base_path.with_name("model_info.json")


@dataclasses.dataclass
class TraningConfig:
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 8
    lr: float = 3e-4
    epochs: int = 1000
    early_stop_patience: int = 50

    trim_paddings_at_end: bool = True
    view_inference_while_training: bool = False
    display_only_wrong_inferences: bool = True

    save_checkpoints: bool = False


@dataclasses.dataclass
class OnnxConfig:
    pytorch_model_path: Path
    version: tuple[int, int, int] = (1, 0, 0)


@dataclasses.dataclass
class MiscConfig:
    start_timestamp = datetime.now(timezone.utc).astimezone()

    @property
    def start_timestamp_str(self):
        return self.start_timestamp.strftime("%Y%m%d_%H%M%S")
