from pathlib import Path
from typing import Optional

from peewee import CharField, Database, DoesNotExist, Model

import config as cfg


class ImageFile(Model):
    filename = CharField(primary_key=True, unique=True)
    label = CharField()

    class Meta:
        table_name = "image_files"


class DatabaseHelper:
    def __init__(self, db: Database = cfg.LABELS.database):
        self.db = db
        self.db.bind([ImageFile])
        self.db.create_tables([ImageFile])

    def classify_sample(self, filepath: Path, label: str):
        ImageFile.insert(
            filename=filepath.name, label=label
        ).on_conflict_replace().execute()
        self.db.commit()

    def skip_sample(self, filepath: Path):
        self.classify_sample(filepath, "__SKIPPED__")

    def remove_sample(self, filepath: Path):
        try:
            item = ImageFile.get(ImageFile.filename == filepath.name)
            item.delete_instance()
        except DoesNotExist:
            pass

    def get_sample_label(
        self, filepath: Path, ignore_skipped: bool = True
    ) -> Optional[str]:
        try:
            item = ImageFile.get(ImageFile.filename == filepath.name)
            if ignore_skipped and item.label == "__SKIPPED__":
                return None
            return item.label
        except DoesNotExist:
            return None

    def count(self):
        return ImageFile.select().count()
