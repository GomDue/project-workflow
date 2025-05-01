from datetime import datetime

NOW_TIME = datetime.now().strftime('%y%m%d')


class YoloModel:
    """
    YOLO 모델을 로드하고 설정에 따라 학습 및 저장하는 클래스.

    - 파라미터 파일(.yaml)에서 학습 하이퍼파라미터를 읽어와 모델을 학습
    - TensorBoard 로그 및 모델 가중치 저장 지원
    - 주로 객체 탐지 모델 학습에 사용

    Args:
        class_path (str): 분리배출 분류 클래스가 정의된 데이터 yaml 경로
        param_path (str): 모델 설정 및 하이퍼파라미터가 정의된 yaml 파일 경로

    Example:
        model = YoloModel("recycle.yaml", "params.yaml")
        model.train()
    """

    def __init__(
        self,
        class_path: str,
        param_path: str,
    ) -> None:
        import yaml
        from ultralytics import YOLO, settings

        self.class_path = class_path

        # Load training parameters
        with open(param_path) as f:
            self.params = yaml.safe_load(f)["yolo"]

        # Load YOLO model with pretrained weights
        self.model = YOLO(self.params["model"])

        # YOLO global settings
        settings.update({
            "runs_dir": self.params["runs_dir"],  # where training logs are saved
            "mlflow": False,
            "tensorboard": True,
        })

    def train(self):
        """
        YOLO 모델을 학습하고 검증한 뒤, state_dict 형태로 가중치를 저장합니다.

        - 학습 로그는 TensorBoard에 기록됩니다.
        - 학습된 모델은 지정된 디렉토리에 저장됩니다.
        - 모델 성능 검증도 함께 수행됩니다.
        """
        self.model.train(
            data=self.class_path,
            epochs=self.params["epochs"],
            patience=self.params["patience"],
            batch=self.params["batch"],
            imgsz=self.params["imgsz"],
            cache=self.params["cache"],
            device=self.params["device"],
            workers=self.params["workers"],
            name=self.params["name"] + f"_{NOW_TIME}",
            pretrained=self.params["pretrained"],
            optimizer=self.params["optimizer"],
            verbose=self.params["verbose"],
            lr0=self.params["lr0"],
            momentum=self.params["momentum"],
            weight_decay=self.params["weight_decay"],
            box=self.params["box"],
            cls=self.params["cls"],
            dropout=self.params["dropout"],
            val=self.params["val"],
            project=self.params["save_dir"] + f"{NOW_TIME}"
        )

        # Validation
        self.model.val()

        # Save model weights (state_dict)
        import torch
        torch.save(self.model.state_dict(), self.params["save_state_dict_dir"] + f"{NOW_TIME}.pt")
