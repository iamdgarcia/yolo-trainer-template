from yolo_cin import CINTrainer
from ultralytics import YOLO
def main():
    # trainer = CINTrainer(cfg="./assets/config.yaml")
    # trainer.train()

    model = YOLO()
    model.train(cfg="./assets/config.yaml")

if __name__ == "__main__":
    main()