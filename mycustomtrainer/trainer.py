from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import de_parallel
from mycustomtrainer.dataset import MyCustomDataset

class MyCustomTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        dataset = MyCustomDataset(data=self.data,img_path=img_path, batch_size=batch, augment=mode == "train",rect=mode == "val", stride=gs)
        return dataset
    