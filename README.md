# Custom Ultralytics Trainer for Slice-Labeled Datasets

Welcome to the **Custom Ultralytics Trainer** repository! This project provides a template for creating a custom YOLO trainer tailored for slice-labeled datasets. The code eliminates image resizing and integrates a modular pipeline for augmentations, dataset handling, and training.

This repository is built with **Poetry** for dependency management, and training scripts are provided for ease of use.

---

## Features

- **Custom Augmentations**: Augmentation pipeline designed to preserve image resolution.
- **Custom Dataset Loader**: Tailored for slice-labeled datasets with flexibility to adapt to specific formats.
- **Custom Trainer**: Extends the Ultralytics YOLO trainer for seamless integration of modifications.
- **Training Script**: A reusable training script located in `scripts/train`.

---

## Project Structure

```
├── mycustomtrainer/
│   ├── augment.py       # Defines the custom augmentation pipeline
│   ├── dataset.py       # Implements the custom dataset loader
│   ├── trainer.py       # Contains the custom trainer class
├── scripts/
│   ├── train.py         # Training script to execute model training
├── README.md            # Documentation for the project
├── pyproject.toml       # Poetry configuration file
├── poetry.lock          # Poetry lock file for dependencies
└── .gitignore           # Ignore unnecessary files and folders
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Poetry installed on your machine. Install it using:

```bash
pip install poetry
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/iamdgarcia/custom-ultralytics-trainer.git
cd custom-ultralytics-trainer
```

2. Install dependencies using Poetry:

```bash
poetry install
```

---

## How to Use

### 1. Modify the Augmentations

Customize the augmentation pipeline by editing `mycustomtrainer/augment.py`. Example augmentations like horizontal flips and brightness adjustments are already provided.

### 2. Define Your Dataset

Adjust `mycustomtrainer/dataset.py` to handle your slice-labeled dataset. This file includes logic to bypass resizing and manage slice-specific image loading.

### 3. Train Your Model

Use the provided training script in `scripts/train.py` to start the training process:

```bash
poetry run python scripts/train.py --cfg your_config.yaml
```

---

## Training Script Overview

The `scripts/train.py` script simplifies the training process by leveraging the custom trainer, dataset, and augmentation pipeline. You only need to provide the configuration file.

**Example usage**:

```bash
poetry run python scripts/train.py --cfg configs/custom_config.yaml
```

**Script highlights**:
- Automatically uses the `CustomTrainer` class from `mycustomtrainer/trainer.py`.
- Parses command-line arguments for flexibility.
- Supports additional logging or callbacks.

---

## Examples

### Example Augmentation

```python
from ultralytics.yolo.data.augment import Albumentations
import albumentations as A

def custom_augmentations():
    return Albumentations(transforms=[
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
```

### Example Dataset

```python
from ultralytics.yolo.data.dataset import YOLODataset

class SliceDataset(YOLODataset):
    def __init__(self, img_paths, label_paths, img_size=None, augment=False):
        super().__init__(img_paths, label_paths, img_size, augment)
        self.img_size = img_size  # Disable resizing

    def load_image(self, index):
        img = super().load_image(index)
        return img
```

### Example Trainer

```python
from ultralytics.yolo.engine.trainer import BaseTrainer

class CustomTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dataset_class = SliceDataset

    def get_dataloader(self, split):
        dataloader = super().get_dataloader(split)
        dataloader.dataset.transforms = custom_augmentations()
        return dataloader
```

---

## Contributing

We welcome contributions! Feel free to submit issues or pull requests to enhance this template.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for their modular and powerful object detection framework.
- [Albumentations](https://albumentations.ai) for the amazing image augmentation library.

---

## Contact

For any questions or suggestions, please reach out to:

- **GitHub**: [@iamdgarcia](https://github.com/iamdgarcia)
- **Email**: info@iamdgarcia.com


