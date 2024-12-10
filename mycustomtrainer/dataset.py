from ultralytics.data.augment import Compose, LetterBox, RandomHSV,Format, RandomPerspective, CopyPaste,MixUp,Albumentations,RandomFlip,RandomFlip
from copy import deepcopy
from ultralytics.data.dataset import YOLODataset
from mycustomtrainer.augment import MyCustomMosaic
import numpy as np
from pathlib import Path
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
import cv2 as cv2
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

            affine = RandomPerspective(
                degrees=hyp.degrees,
                translate=hyp.translate,
                scale=0,
                shear=hyp.shear,
                perspective=hyp.perspective,
                )

            pre_transform = Compose([mosaic,affine])
            pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))

            transforms = Compose(
                [
                    pre_transform,
                    MixUp(self, pre_transform=pre_transform, p=hyp.mixup),
                    Albumentations(p=1.0),
                    RandomFlip(direction="vertical", p=hyp.flipud),
                    RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=[]),
                ])
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms
    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)

                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f, cv2.IMREAD_UNCHANGED)  # BGR
            else:  # read image
                im = cv2.imread(f, cv2.IMREAD_UNCHANGED)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            if im.ndim == 2:  # Grayscale image
                im = np.expand_dims(im, axis=-1)
            return im, (h0, w0), im.shape[:2]
        if self.ims[i].ndim == 2:  # Grayscale image
            self.ims[i] = np.expand_dims(self.ims[i], axis=-1)
        return self.ims[i], self.im_hw0[i], self.im_hw[i]