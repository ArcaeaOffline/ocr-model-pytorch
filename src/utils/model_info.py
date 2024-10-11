import dataclasses
from datetime import datetime, timezone

import config as cfg


@dataclasses.dataclass
class ModelInfo:
    image_width: int
    image_height: int
    labels: list[str]
    dims: int
    grayscale: bool
    use_attention: bool
    use_ctc: bool

    built_time: datetime

    blank_token: str = cfg.MODEL.blank_token
    pad_token: str = cfg.MODEL.pad_token

    @property
    def built_timestamp(self) -> int:
        return int(self.built_time.astimezone(timezone.utc).timestamp())

    def asdict(self):
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "labels": self.labels.copy(),
            "dims": self.dims,
            "grayscale": self.grayscale,
            "use_attention": self.use_attention,
            "use_ctc": self.use_ctc,
            "blank_token": self.blank_token,
            "pad_token": self.pad_token,
            "built_timestamp": self.built_timestamp,
        }

    @classmethod
    def from_dict(cls, model_info_dict: dict):
        return cls(
            image_width=model_info_dict["image_width"],
            image_height=model_info_dict["image_height"],
            labels=model_info_dict["labels"].copy(),
            dims=model_info_dict["dims"],
            grayscale=model_info_dict["grayscale"],
            use_attention=model_info_dict["use_attention"],
            use_ctc=model_info_dict["use_ctc"],
            built_time=datetime.fromtimestamp(
                model_info_dict["built_timestamp"], timezone.utc
            ),
            blank_token=model_info_dict["blank_token"],
            pad_token=model_info_dict["pad_token"],
        )


def dump_model_info(labels: list[str]) -> ModelInfo:
    return ModelInfo(
        image_width=cfg.MODEL.image_width,
        image_height=cfg.MODEL.image_height,
        labels=labels.copy(),
        dims=cfg.MODEL.dims,
        grayscale=cfg.MODEL.grayscale,
        use_attention=cfg.MODEL.use_attention,
        use_ctc=cfg.MODEL.use_ctc,
        built_time=cfg.MISC.start_timestamp,
        blank_token=cfg.MODEL.blank_token,
        pad_token=cfg.MODEL.pad_token,
    )
