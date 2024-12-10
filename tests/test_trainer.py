import pytest
from yolo_cin import CINTrainer

def test_standard_execution():
    """
    Verifica que el entrenamiento se ejecuta sin errores.
    """
    try:
        from ultralytics import YOLO
        model = YOLO()  # no YOLO11n-world model yet
        model.train(
            data="coco8.yaml",
            epochs=1,
            imgsz=32,
            cache="disk",
            close_mosaic=1,
        )
    except Exception as e:
        pytest.fail(f"El entrenamiento falló con la siguiente excepción: {e}")

def test_cin_trainer_execution():
    """
    Verifica que el entrenamiento se ejecuta sin errores.
    """
    try:
        from ultralytics import YOLO
        model = YOLO()  # no YOLO11n-world model yet
        model.train(
            data="dota8.yaml",
            epochs=1,
            imgsz=32,
            cache="disk",
            close_mosaic=1,
            trainer=CINTrainer
        )
    except Exception as e:
        pytest.fail(f"El entrenamiento falló con la siguiente excepción: {e}")
