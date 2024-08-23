from pathlib import Path

import config as cfg


class SamplesHelper:
    def __init__(self, directory: Path = cfg.SAMPLES.path):
        self.directory = directory
        self.samples: list[Path] = []

        self.refresh_samples()

    def refresh_samples(self):
        self.samples.clear()
        for glb in cfg.SAMPLES.globs:
            self.samples.extend(self.directory.glob(glb))
        self.samples = sorted(self.samples, reverse=True)
