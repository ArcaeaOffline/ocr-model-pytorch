import dataclasses
from datetime import datetime, timezone

import config as cfg


@dataclasses.dataclass
class ModelInfo:
    image_width: int
    image_height: int
    classes: list[str]
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
            "classes": self.classes.copy(),
            "built_timestamp": self.built_timestamp,
            "blank_token": self.blank_token,
            "pad_token": self.pad_token,
        }

    @classmethod
    def from_dict(cls, model_info_dict: dict):
        return cls(
            image_width=model_info_dict["image_width"],
            image_height=model_info_dict["image_height"],
            classes=model_info_dict["classes"].copy(),
            built_time=datetime.fromtimestamp(
                model_info_dict["built_timestamp"], timezone.utc
            ),
            blank_token=model_info_dict["blank_token"],
            pad_token=model_info_dict["pad_token"],
        )


def dump_model_info(classes: list[str]) -> ModelInfo:
    return ModelInfo(
        image_width=cfg.MODEL.image_width,
        image_height=cfg.MODEL.image_height,
        classes=classes.copy(),
        built_time=cfg.MISC.start_timestamp,
        blank_token=cfg.MODEL.blank_token,
        pad_token=cfg.MODEL.pad_token,
    )
