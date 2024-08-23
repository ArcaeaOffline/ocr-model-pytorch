import os
from pathlib import Path

import numpy as np
import PIL.ImageFile
import torch.utils.data
from PIL import Image
from sklearn import model_selection, preprocessing

import config as cfg
from src.utils import DatabaseHelper, SamplesHelper

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations  # noqa: E402

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset:
    def __init__(self, image_paths: list[Path], targets, resize=None, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        targets = self.targets[item]

        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]),
                resample=Image.Resampling.BILINEAR,
            )

        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)

        return {
            "images": image,
            "targets": torch.tensor(targets, dtype=torch.long),
        }


def build_dataloaders(
    database_helper: DatabaseHelper = DatabaseHelper(),
    samples_helper: SamplesHelper = SamplesHelper(),
):
    sample_files = []
    original_targets: list[str] = []

    samples_helper.refresh_samples()
    for sample in samples_helper.samples:
        label = database_helper.get_sample_label(sample, ignore_skipped=True)
        if label is None:
            continue
        sample_files.append(sample)
        original_targets.append(label)

    label_max_width = 0
    if cfg.LABELS.max_width > 0:
        label_max_width = cfg.LABELS.max_width
    else:
        label_max_width = max([len(x) for x in original_targets])
    padded_targets = [target.ljust(label_max_width, "-") for target in original_targets]
    # padded_targets = original_targets
    targets = [[c for c in x] for x in padded_targets]
    targets_flat = [c for clist in targets for c in clist]
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(targets_flat)
    targets_encoded = [label_encoder.transform(x) for x in targets]
    targets_encoded = np.array(targets_encoded)
    targets_encoded = targets_encoded + 1

    (train_imgs, test_imgs, train_targets, test_targets, _, test_original_targets) = (
        model_selection.train_test_split(
            sample_files,
            targets_encoded,
            original_targets,
            test_size=0.1,
            random_state=616,
        )
    )

    train_dataset = ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(cfg.MODEL.image_height, cfg.MODEL.image_width),
    )
    # train_dataset_normalize = ClassificationDataset(
    #     image_paths=train_imgs,
    #     targets=train_targets,
    #     resize=(cfg.MODEL.image_height, cfg.MODEL.image_width),
    #     transform=albumentations.Compose([albumentations.Normalize(always_apply=True)]),
    # )
    train_dataset_grayscale = ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(cfg.MODEL.image_height, cfg.MODEL.image_width),
        transform=albumentations.Compose([albumentations.ToGray(always_apply=True)]),
    )
    train_dataset_downscale = ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(cfg.MODEL.image_height, cfg.MODEL.image_width),
        transform=albumentations.Compose(
            [albumentations.Downscale(scale_range=(0.4, 0.4), always_apply=True)]
        ),
    )

    train_dataset_merged = torch.utils.data.ConcatDataset(
        [train_dataset, train_dataset_grayscale, train_dataset_downscale]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset_merged,
        batch_size=cfg.TRAINING.batch_size,
        num_workers=cfg.TRAINING.num_workers,
        shuffle=True,
    )

    test_dataset = ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(cfg.MODEL.image_height, cfg.MODEL.image_width),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TRAINING.batch_size,
        num_workers=cfg.TRAINING.num_workers,
        shuffle=False,
    )
    return train_loader, test_loader, test_original_targets, label_encoder.classes_
