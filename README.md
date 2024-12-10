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
class MyCustomMosaic(BaseMixTransform):
    def __init__(self, dataset, imgsz, hyp, mosaic_scale=(0.5, 1.5), mosaic_prob=1.0, n=4):
        """
        Custom mosaic transformation that places images onto a grid-based canvas without overlapping.

        Args:
            dataset (Dataset): The dataset instance to access images and labels.
            imgsz (int): The target image size.
            hyp (dict): Hyperparameters for augmentation.
            mosaic_scale (tuple): Scale range for resizing images before placement.
            mosaic_prob (float): Probability of applying the mosaic transform.
            n (int): Number of images in the mosaic (4 or 9).
        """
        assert n in {4, 9}, "The number of images 'n' must be either 4 or 9."
        super().__init__(dataset=dataset, p=mosaic_prob)
        self.imgsz = imgsz
        self.hyp = hyp
        self.mosaic_scale = mosaic_scale
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height

        self.n = n  # Number of images in the mosaic
```

### Example Dataset

```python
class MyCustomDataset(YOLODataset):
    """
    Custom Dataset class that uses a mosaic canvas as the starting point and randomly
    places images onto it before applying other transformations.
    """

    def build_transforms(self, hyp=None):
        """Builds and appends custom transforms to the list."""
        if self.augment:
            mosaic =MyCustomMosaic(
                        self,
                        self.imgsz,
                        hyp,
                        mosaic_scale=(0.5, 1.5),
                        mosaic_prob=hyp.mosaic,
                        n=9  # Change to 9 if you want a 9-image mosaic
                    )
```

### Example Trainer

```python
class MyCustomTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        dataset = MyCustomDataset(data=self.data,img_path=img_path, batch_size=batch, augment=mode == "train",rect=mode == "val", stride=gs)
        return dataset
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


