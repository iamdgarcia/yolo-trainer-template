from ultralytics.data.augment import BaseMixTransform
import numpy as np
import random
import cv2
from ultralytics.utils.instance import Instances

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

    def get_indexes(self):
        """Returns a list of random indexes from the dataset for mosaic augmentation."""
        return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n)]  # n - 1 additional images

    def _mix_transform(self, labels):
        """Applies mosaic augmentation to the input image and labels."""
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augmentation."
        if self.n == 9:
            return self._mosaic9(labels)
        else:
            raise ValueError("Invalid number of images for mosaic. Choose either 4 or 9.")

    def _mosaic9(self, labels):
        """Creates a 3x3 mosaic using 9 images, randomly cropped to fit the final size of (imgsz, imgsz)."""
        s = self.imgsz
        img = labels['img']
        fill_value = np.iinfo(img.dtype).max * 114 // 255  # Adjust fill value based on data type
        # Initialize mosaic image with target size (s, s)
        img9 = np.full((s, s, img.shape[2]), fill_value, dtype=img.dtype)
        mosaic_labels = []

        # Scale each grid cell to one-third of the final size
        cell_size = s // 3
        for i in range(9):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]

            img = labels_patch['img']
            h, w = img.shape[:2]

            # Randomly crop image to cell size
            if h > cell_size or w > cell_size:
                crop_y1 = np.random.randint(0, max(1, h - cell_size + 1))
                crop_x1 = np.random.randint(0, max(1, w - cell_size + 1))
                img = img[crop_y1:crop_y1 + cell_size, crop_x1:crop_x1 + cell_size]

            h, w = img.shape[:2]  # Update dimensions after cropping

            # Place img in img9
            if i == 0:  # center
                c = (s // 2) - (w // 2), (s // 2) - (h // 2), (s // 2) + (w // 2), (s // 2) + (h // 2)
            elif i == 1:  # top
                c = (s // 2) - (w // 2), 0, (s // 2) + (w // 2), h
            elif i == 2:  # top right
                c = s - w, 0, s, h
            elif i == 3:  # right
                c = s - w, (s // 2) - (h // 2), s, (s // 2) + (h // 2)
            elif i == 4:  # bottom right
                c = s - w, s - h, s, s
            elif i == 5:  # bottom
                c = (s // 2) - (w // 2), s - h, (s // 2) + (w // 2), s
            elif i == 6:  # bottom left
                c = 0, s - h, w, s
            elif i == 7:  # left
                c = 0, (s // 2) - (h // 2), w, (s // 2) + (h // 2)
            elif i == 8:  # top left
                c = 0, 0, w, h

            # Compute target region
            x1, y1, x2, y2 = (max(0, min(x, s)) for x in c)  # Clamp to grid size

            # Adjust source and target slices
            target_h = y2 - y1
            target_w = x2 - x1
            img9[y1:y1 + target_h, x1:x1 + target_w] = img[:target_h, :target_w]

            # Update labels
            padw, padh = c[:2]  # Offset for label adjustment
            labels_patch = self._update_labels(labels_patch, padw-crop_x1, padh-crop_y1)
            mosaic_labels.append(labels_patch)

        # Concatenate all labels and add the mosaic image
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img9
        return final_labels

    
    @staticmethod
    def _update_labels(labels, padw, padh):
        """
        Updates label coordinates with padding values.

        This method adjusts the bounding box coordinates of object instances in the labels by adding padding
        values. It also denormalizes the coordinates if they were previously normalized.

        Args:
            labels (Dict): A dictionary containing image and instance information.
            padw (int): Padding width to be added to the x-coordinates.
            padh (int): Padding height to be added to the y-coordinates.

        Returns:
            (Dict): Updated labels dictionary with adjusted instance coordinates.

        Examples:
            >>> labels = {"img": np.zeros((100, 100, 3)), "instances": Instances(...)}
            >>> padw, padh = 50, 50
            >>> updated_labels = Mosaic._update_labels(labels, padw, padh)
        """
        nh, nw = labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels
    def _cat_labels(self, mosaic_labels):
        """
        Concatenates and processes labels for mosaic augmentation.

        This method combines labels from multiple images used in mosaic augmentation, clips instances to the
        mosaic border, and removes zero-area boxes.

        Args:
            mosaic_labels (List[Dict]): A list of label dictionaries for each image in the mosaic.

        Returns:
            (Dict): A dictionary containing concatenated and processed labels for the mosaic image, including:
                - im_file (str): File path of the first image in the mosaic.
                - ori_shape (Tuple[int, int]): Original shape of the first image.
                - resized_shape (Tuple[int, int]): Shape of the mosaic image (imgsz * 2, imgsz * 2).
                - cls (np.ndarray): Concatenated class labels.
                - instances (Instances): Concatenated instance annotations.
                - mosaic_border (Tuple[int, int]): Mosaic border size.
                - texts (List[str], optional): Text labels if present in the original labels.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640)
            >>> mosaic_labels = [{"cls": np.array([0, 1]), "instances": Instances(...)} for _ in range(4)]
            >>> result = mosaic._cat_labels(mosaic_labels)
            >>> print(result.keys())
            dict_keys(['im_file', 'ori_shape', 'resized_shape', 'cls', 'instances', 'mosaic_border'])
        """
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        # Final labels
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": [0,0],
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels