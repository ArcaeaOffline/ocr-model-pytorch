# Pytorch CRNN OCR model

> This is a fork of [GabrielDornelles/pytorch-ocr](https://github.com/GabrielDornelles/pytorch-ocr).

## What's changed

* Use dataclasses for config
* Use SQLite database for labelling (with a crappy GUI classifier)
* Experimental ONNX support
* Limited Tensorboard logging support
* Removed some fancy log messages during training

## TODOs?

* Quantization?

## License

> The original project, [pytorch-ocr](https://github.com/GabrielDornelles/pytorch-ocr), is licensed under the MIT license, Copyright (c) 2021 Nanohana. See [LICENSE](./LICENSE) for details.

Copyright (c) 2024 Lin He

This file is part of Pytorch CRNN OCR model, along with all the accessories included in the project, such as samples classifier, dataset loader, model trainer and so on (all referred as "this software" below).

This software is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this software. If not, see <https://www.gnu.org/licenses/>.
